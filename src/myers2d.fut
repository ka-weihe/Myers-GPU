let myers [n][m] (a: [n]u16) (b: [m]u16) =
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
    let xv = eq | mv
    let xh = (((eq & pv) + pv) ^ pv) | eq
    let ph = mv | ~(xh | pv)
    let mh = pv & xh
    let sc = if bool.u64 (ph & lst) then sc + 1 else (
      if bool.u64 (mh & lst) then sc - 1 else sc)
    let ph = (ph << 1) | 1
    let pv = (mh << 1) | ~(xv | ph)
    let mv = ph & xv
    in (sc, mv, pv)).1

-- myers
-- ==
-- entry: _myers
-- input { [1u16] [1u16] } output { 0u16 }
-- input { [1u16, 2u16, 3u16] [1u16, 2u16, 4u16] } output { 1u16 }
-- input { [1u16, 2u16, 3u16] [3u16, 2u16, 1u16] } output { 2u16 }
-- input { [1u16, 2u16, 3u16] [4u16, 5u16, 6u16] } output { 3u16 }
-- input { [1u16, 2u16, 3u16, 4u16] [1u16] } output { 3u16 }

entry _myers = myers

let myers_no_peq [m] (n: i32) (peq: []u64) (lst: u64) (b: [m]u16) =
  let mv = 0u64
  let pv = -1u64
  let sc = u16.i32 n

  in (loop (sc, mv, pv) for bc in b do
    let eq = unsafe peq[i32.u16 bc]
    let xv = eq | mv
    let xh = (((eq & pv) + pv) ^ pv) | eq
    let ph = mv | ~(xh | pv)
    let mh = pv & xh
    let sc = if bool.u64 (ph & lst) then sc + 1 else (
      if bool.u64 (mh & lst) then sc - 1 else sc)
    let ph = ((ph << 1) | 1)
    let pv = (mh << 1) | ~(xv | ph)
    let mv = ph & xv
    in (sc, mv, pv)).1

let myers_many_2d [n][m][o] (a: [n]u16) (bs: [m][o]u16) =
  let peq = replicate 256 0u64
  let lst = 1u64 << (u64.i32 n - 1u64)
  let peq = loop peq for i < n do
    let j = i32.u16 a[i]
    in unsafe peq with [j] = peq[j] | (1u64 << (u64.i32 i))
  in map (myers_no_peq n peq lst) bs

-- myers_many_2d
-- ==
-- entry: _myers_many_2d
-- input @../datasets/b0to255-64-1000000x64-u16
-- input @../datasets/b0to255-100-1000000x10-u16
-- input @../datasets/b0to255-10-10000000x10-u16
-- input @../datasets/b0to255-10-1000000x10-u16
-- input @../datasets/b0to255-10-100000x10-u16
-- input @../datasets/b0to255-1024-5000x1024-u16
-- input @../datasets/b0to255-512-5000x512-u16
-- input @../datasets/b0to255-256-5000x256-u16
-- input @../datasets/b0to255-128-5000x128-u16
-- input @../datasets/b0to255-64-5000x64-u16
-- input @../datasets/b0to255-32-5000x32-u16
-- input @../datasets/b0to255-16-5000x16-u16

entry _myers_many_2d = myers_many_2d
