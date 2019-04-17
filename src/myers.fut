

-- n = antallet af chars i input strengen
-- m = antallet af chars i alle strengene
-- o = antallet af strenge
-- p = antallet af chars i en enkel streng

-- bs = liste af bokstaver for strenge af længde o
-- hvor index 0 - o er forbokstaverne, o - o * 2 er
-- bokstav 2 for alle strengene osv.

let myers_many [n][m] (a: [n]u16) (bs: [m]u16) (o: i32) =
  let lst = 1u64 << (u64.i32 n - 1u64)
  let scs = replicate o (u16.i32 n)
  let mvs = replicate o 0u64
  let pvs = replicate o (-1u64)
  let peq = replicate 256 0u64
  let ids = iota o

  let peq = loop xs = peq for i < n do
    let j = i32.u16 a[i]
    in xs with [j] = xs[j] | (1u64 << u64.i32 i)

  let result = loop (scs, mvs, pvs) for i in 0..o..<m do
    let ccs = map1 (\ id -> unsafe i32.u16 bs[i + id]) ids
    let eqs = map1 (\ cc -> unsafe peq[cc]) ccs
    let xvs = map2 (\ eq mv -> eq | mv) eqs mvs
    let xhs = map2 (\ eq pv -> (((eq & pv) + pv) ^ pv) | eq) eqs pvs
    let phs = map3 (\ mv xh pv -> mv | ~(xh | pv)) mvs xhs pvs
    let mhs = map2 (\ pv xh -> pv & xh) pvs xhs
    let scs = map3 (\ sc ph mh -> if bool.u64 (ph & lst) then sc + 1 else (
      if bool.u64 (mh & lst) then sc - 1 else sc)) scs phs mhs
    let phs = map (\ ph -> (ph << 1) | 1) phs
    let pvs = map3 (\ mh xv ph -> (mh << 1) | ~(xv | ph)) mhs xvs phs
    let mvs = map2 (\ ph xv -> ph & xv) phs xvs
    in (scs, mvs, pvs)
  in result.1

-- myers_many
-- ==
-- entry: _myers_many
-- input @../datasets/data10x100x1000000
-- input @../datasets/data10x10x10000000
-- input @../datasets/data10x10x1000000
-- input @../datasets/data10x10x100000
-- input @../datasets/data32x32x5000

entry _myers_many = myers_many
