let hyyro [n][m] (a: [n]u16) (b: [m]u16) =
  let mv = 0u64
  let pv = -1u64
  let sc = u16.i32 n
  let lst = 1u64 << (u64.i32 n - 1u64)

  let peq = replicate 256 0u64
  let peq = loop peq for i < n do
    let index = i32.u16 a[i]
    in unsafe peq with [index] = peq[index] | (1u64 << (u64.i32 i))

  in (loop (sc, mv, pv) for i < m do
    let eq = unsafe peq[i32.u16 b[i]]
    let xh = (((eq & pv) + pv) ^ pv) | eq | mv
    let ph = mv | ~(xh | pv)
    let mh = xh & pv
    let sc = if bool.u64 (ph & lst) then sc + 1 else (
      if bool.u64 (mh & lst) then sc - 1 else sc)
    let ph = ((ph << 1) | 1)
    let pv = (mh << 1) | ~(xh | ph)
    let mv = xh & ph
    in (sc, mv, pv)).1

-- hyyro
-- ==
-- entry: _hyyro
-- input { [1u16] [1u16] } output { 0u16 }
-- input { [1u16, 2u16, 3u16] [1u16, 2u16, 4u16] } output { 1u16 }
-- input { [1u16, 2u16, 3u16] [3u16, 2u16, 1u16] } output { 2u16 }
-- input { [1u16, 2u16, 3u16] [4u16, 5u16, 6u16] } output { 3u16 }
-- input { [1u16, 2u16, 3u16, 4u16] [1u16] } output { 3u16 }

entry _hyyro = hyyro
