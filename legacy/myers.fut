

-- n = antallet af chars i input strengen
-- m = antallet af chars i alle strengene
-- o = antallet af strenge
-- p = antallet af chars i en enkel streng

-- Test
-- ==
-- input @data

let main [n][m] (a: [n]u16) (bs: [m]u16) (o: i32) =
  let p = m / o
  let scs = replicate o (u16.i32 n) -- liste af n'er
  let mvs = replicate o 0u64  -- liste af 0er
  let pvs = replicate o (-1u64) -- liste af -1er
  let last = 1u64 << (u64.i32 n - 1u64)
  let range = iota o

  let peq = replicate 256 0u64
  let peq = loop peq for i < n do
    let index = i32.u16 a[i]
    in peq with [index] = peq[index] | (1u64 << (u64.i32 i))

  in (loop (scs, mvs, pvs) for i < p do
    --let chs = map (\ j -> unsafe ) range
    let eqs = map (\ j -> unsafe peq[i32.u16 bs[i + j * p]]) range
    -- let eqs = map (\ j -> unsafe peq[i32.u16 bs[i * o + j]]) range
    let xvs = map2 (\ eq mv -> eq | mv) eqs mvs
    let xhs = map2 (\ eq pv -> (((eq & pv) + pv) ^ pv) | eq) eqs pvs
    let phs = map3 (\ mv xh pv -> mv | ~(xh | pv)) mvs xhs pvs
    let mhs = map2 (\ pv xh -> pv & xh) pvs xhs
    let scs = map3 (\ sc ph mh -> if bool.u64 (ph & last) then sc + 1 else (
      if bool.u64 (mh & last) then sc - 1 else sc)) scs phs mhs
    let phs = map (\ ph -> (ph << 1) | 1) phs
    let pvs = map3 (\ mh xv ph -> (mh << 1) | ~(xv | ph)) mhs xvs phs
    let mvs = map2 (\ ph xv -> ph & xv) phs xvs
    in (scs, mvs, pvs)).1


-- let xvs = map (|) eqs mvs
-- let mhs = map (&) pvs xhs
-- let mvs = map (&) phs xvs
