from max.gpu import global_idx
from max.gpu.host import DeviceContext, HostBuffer
from std.collections.string import atol
from std.math import ceildiv, isfinite, round
from std.os import SEEK_END
from std.sys import argv
from std.time import perf_counter_ns


comptime capacity = UInt64(1) << 32
comptime masked = UInt64(0xFFFF_FFFF_FFFF_FFFF)


def mass_units(value: Float32) -> UInt64:
    return UInt64(round(value * Float32(capacity)))


def pack_grid(
    source: Pointer[Float32, ImmutAnyOrigin],
    destination: Pointer[UInt64, MutAnyOrigin],
    count: UInt32,
):
    var index = UInt32(global_idx.x)
    if index < count:
        var value = source[unsafe_offset=Int(index)]
        destination[
            unsafe_offset=Int(index)
        ] = masked if value < 0 else mass_units(value)


def unpack_grid(
    source: Pointer[UInt64, ImmutAnyOrigin],
    destination: Pointer[Float32, MutAnyOrigin],
    count: UInt32,
):
    var index = UInt32(global_idx.x)
    if index < count:
        var value = source[unsafe_offset=Int(index)]
        destination[
            unsafe_offset=Int(index)
        ] = -1 if value == masked else Float32(value) / Float32(capacity)


def outgoing_share(mass: UInt64) -> UInt64:
    return (mass - capacity) >> 2 if mass > capacity else 0


def neighbour_share(neighbour: UInt64, own_share: UInt64) -> UInt64:
    # A masked neighbour reflects the outgoing share back to its source.
    return own_share if neighbour == masked else outgoing_share(neighbour)


def gather(
    old: Pointer[UInt64, ImmutAnyOrigin],
    next_grid: Pointer[UInt64, MutAnyOrigin],
    width: UInt32,
    height: UInt32,
    wrap_x: UInt8,
):
    var x = UInt32(global_idx.x)
    var y = UInt32(global_idx.y)
    if x >= width or y >= height:
        return

    var index = y * width + x
    var mass = old[unsafe_offset=Int(index)]
    if mass == masked:
        next_grid[unsafe_offset=Int(index)] = masked
        return

    var share = outgoing_share(mass)
    var west = share
    var east = share
    var north = share
    var south = share
    if x > 0:
        west = neighbour_share(old[unsafe_offset=Int(index - 1)], share)
    elif wrap_x != 0:
        west = neighbour_share(old[unsafe_offset=Int(index + width - 1)], share)
    if x + 1 < width:
        east = neighbour_share(old[unsafe_offset=Int(index + 1)], share)
    elif wrap_x != 0:
        east = neighbour_share(
            old[unsafe_offset=Int(index - (width - 1))], share
        )
    if y > 0:
        north = neighbour_share(old[unsafe_offset=Int(index - width)], share)
    if y + 1 < height:
        south = neighbour_share(old[unsafe_offset=Int(index + width)], share)

    # Retain the indivisible remainder at the donor to conserve integer mass.
    next_grid[unsafe_offset=Int(index)] = (
        mass - (share << 2) + west + east + north + south
    )


def read_grid(path: StringSpan, mut buffer: HostBuffer[DType.float32]) raises:
    var input = open(path, "r")
    var expected_bytes = len(buffer) * 4
    if input.seek(0, SEEK_END) != UInt64(expected_bytes):
        raise Error(
            "Input byte count does not match WIDTH × HEIGHT float32 cells"
        )
    _ = input.seek(0)
    if input.read(buffer.as_span()) != expected_bytes:
        raise Error("Could not read the entire input grid")
    var total = UInt64(0)
    for mass in buffer.as_span():
        if (
            not isfinite(mass)
            or (mass < 0 and mass != -1)
            or mass >= Float32(capacity)
        ):
            raise Error(
                "Input cells must be between 0 and 2^32 marks (exclusive upper"
                " limit) or -1 for masked cells"
            )
        if mass >= 0:
            var units = mass_units(mass)
            if units >= masked - total:
                raise Error("Total grid mass must be below 2^32 marks")
            total += units


def write_grid(path: StringSpan, buffer: HostBuffer[DType.float32]) raises:
    var output = open(path, "w")
    var bytes = Span(
        unsafe_ptr=buffer.unsafe_ptr().unsafe_bitcast[UInt8](),
        length=len(buffer) * 4,
    )
    output.write_all(bytes)
    output.close()


def main() raises:
    var args = argv()
    if len(args) != 6 and (len(args) != 7 or args[6] != "--wrap-x"):
        raise Error(
            "Usage: poppyn-redistribute WIDTH HEIGHT INPUT OUTPUT PASSES"
            " [--wrap-x]"
        )
    var width = atol(args[1])
    var height = atol(args[2])
    var passes = atol(args[5])
    if width <= 0 or height <= 0 or width > 2147483647 // height or passes < 0:
        raise Error(
            "Use positive dimensions with at most 2,147,483,647 cells and a"
            " non-negative pass count"
        )

    var wrap_x = len(args) == 7
    var count = width * height
    comptime block_width = 32
    comptime block_height = 8
    var grid = (ceildiv(width, block_width), ceildiv(height, block_height))
    var context = DeviceContext()
    var host = context.enqueue_create_host_buffer[DType.float32](count)
    context.synchronize()
    read_grid(args[3], host)
    if passes == 0:
        write_grid(args[4], host)
        return

    var transport = context.enqueue_create_buffer[DType.float32](count)
    var old = context.enqueue_create_buffer[DType.uint64](count)
    var result = context.enqueue_create_buffer[DType.uint64](count)
    context.enqueue_copy(dst_buf=transport, src_buf=host)
    comptime conversion_block_size = block_width * block_height
    var conversion_grid = ceildiv(count, conversion_block_size)
    context.enqueue_function[pack_grid](
        transport.unsafe_ptr().as_imm().as_unsafe_any_origin(),
        old.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(count),
        grid_dim=conversion_grid,
        block_dim=conversion_block_size,
    )
    context.synchronize()
    var start = perf_counter_ns()
    # The context runs commands in stream order. Each pass reads the last pass's
    # completed buffer and owns every write to its other buffer.
    for _ in range(passes):
        context.enqueue_function[gather](
            old.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            result.unsafe_ptr().as_unsafe_any_origin(),
            UInt32(width),
            UInt32(height),
            UInt8(1) if wrap_x else UInt8(0),
            grid_dim=grid,
            block_dim=(block_width, block_height),
        )
        swap(old, result)
    context.synchronize()
    var elapsed = Float64(perf_counter_ns() - start) / 1e6

    context.enqueue_function[unpack_grid](
        old.unsafe_ptr().as_imm().as_unsafe_any_origin(),
        transport.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(count),
        grid_dim=conversion_grid,
        block_dim=conversion_block_size,
    )
    context.enqueue_copy(dst_buf=host, src_buf=transport)
    context.synchronize()
    write_grid(args[4], host)
    print("GPU passes:", passes, "Elapsed ms:", elapsed)
