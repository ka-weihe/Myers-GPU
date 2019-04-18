

-- | Datatype used.
-- type t = u32
--
-- -- | Bignum type.
-- type uinf = []t
--
-- -- | Intermediate calculations type.
-- -- | Must be able to hold larger values than t.
-- type i = u64
--
-- let (x: uinf) | (y: uinf) = map2 (|) x y
-- let (x: uinf) & (y: uinf) = map2 (&) x y
-- let ~ (x: uinf) = map (~) x


let h = u16.u8 u8.highest
let bv_adder (x: []u8) (y: []u8): []u8 =
  let nps = scan (\ (c, _) (a, b) ->
    let r = (u16.u8 a) + (u16.u8 b) + (u16.u8 c)
    in (u8.bool (r > h), u8.u16 r)) (0, 0) (zip x y)
  in map (.2) nps

--& (u16.u8 u8.highest))
--(pairs ++ [(0, 0)])

-- Bitvector adder.
-- ==
-- entry: _bv_adder
-- input { [255u8] [1u8] } output { [0u8, 1u8] }

entry _bv_adder = bv_adder
