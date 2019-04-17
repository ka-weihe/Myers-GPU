import "myers"
import "poly"

let compare [n][m] (a: [n]u16) (bs: [m]u16) (o: i32) =
  let expected = levenshtein_many a bs o
  let actual = myers_many a bs o
  let correct = all (\ (x, y) -> x == y) (zip expected actual)
  in correct

--
-- ==
-- entry: _compare
-- input @../datasets/data10x100x1000000 output {true}
-- input @../datasets/data10x10x10000000 output {true}
-- input @../datasets/data10x10x1000000 output {true}
-- input @../datasets/data10x10x100000 output {true}

entry _compare = compare
