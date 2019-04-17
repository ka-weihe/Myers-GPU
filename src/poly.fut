
-- | Encode to our Myers format.
-- | bs: List of strings.
let encode [n][m] (bs: [n][m]u16): []u16 =
  flatten (transpose bs)

-- encode
-- ==
-- entry: _encode
-- input { [[1u16, 2u16], [3u16, 4u16]] }
-- output { [1u16, 3u16, 2u16, 4u16] }

entry _encode = encode

-- | Decode from our Myers format.
-- | bs: List of formatted strings.
-- | m: Number of strings.
let decode [n] (bs: [n]u16) (m: i32): [][]u16 =
  transpose (unflatten (n / m) m bs)

-- decode
-- ==
-- entry: _decode
-- input { [1u16, 3u16, 2u16, 4u16] 2 }
-- output { [[1u16, 2u16], [3u16, 4u16]] }

entry _decode = decode

-- | Helper function for Levenshtein.
let levenshtein1 [n][m] (a: [n]u16) (b: [m]u16): u16 =
  last (loop r = 1..2..<(u16.i32 m) + 1 for j < n do
    (loop (i, d, r) = (u16.i32 j, u16.i32 j + 1, r) for k < m do
      let t = unsafe r[k]
      let d = if a[j] == b[k] then i
        else 1 + u16.min i (u16.min d t)
      let r = unsafe update r k d
      in (t, d, r)).3)

-- | Calculates the Levenshtein distance.
-- | a & b: The strings to find the distance between.
let levenshtein [n][m] (a: [n]u16) (b: [m]u16): u16 =
  if m > n then levenshtein1 b a else levenshtein1 a b

-- levenshtein
-- ==
-- entry: _levenshtein
-- input { [1u16] [1u16] } output { 0u16 }
-- input { [1u16, 2u16, 3u16] [1u16, 2u16, 4u16] } output { 1u16 }
-- input { [1u16, 2u16, 3u16] [3u16, 2u16, 1u16] } output { 2u16 }
-- input { [1u16, 2u16, 3u16] [4u16, 5u16, 6u16] } output { 3u16 }
-- input { [1u16, 2u16, 3u16, 4u16] [1u16] } output { 3u16 }

entry _levenshtein = levenshtein

-- | Calculates the Levenshtein between a input
-- | string and a list of strings.
-- | a: Input string.
-- | bs: List of formatted strings.
-- | o: Number of strings.
let levenshtein_many [n][m] (a: [n]u16) (bs: [m]u16) (o: i32) =
  let bs = decode bs o
  in map (\ b -> levenshtein a b) bs

-- levenshtein_many
-- ==
-- entry: _levenshtein_many
-- input @../datasets/data10x100x1000000
-- input @../datasets/data10x10x10000000
-- input @../datasets/data10x10x1000000
-- input @../datasets/data10x10x100000
-- input @../datasets/b0to255-10-1000000-100000-u16
-- input @../datasets/data32x32x5000

entry _levenshtein_many = levenshtein_many

-- | Levenshtein without any decoding. Same input.
let levenshtein_many_2d [n][m][o] (a: [n]u16) (bs: [m][o]u16) =
  map (levenshtein a) bs

-- levenshtein_many_2d
-- ==
-- entry: _levenshtein_many_2d
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

entry _levenshtein_many_2d = levenshtein_many_2d
