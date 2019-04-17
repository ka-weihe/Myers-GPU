let nimedit [n][m] (a: [n]u16) (b: [m]u16) =
  let half = n >> 1
  let row1 =  map (\i -> i) (iota (m - half - 1))
  let row = update row1 0 (n - half - 1)

  let n = n + 1
  let m = m + 1
  loop for i in 1..(n - 1) do
    let char1 = a[i - 1]

    if i >= (n - half) then
      let offset = i - n + half
      let char2p = offset
      let p = offset

      let c3 = row[p] + u16.bool (char1 != b[char2p])
      let p = p + 1
      let char2p = char2p + 1
      let x = row[p] + 1
      let D = x
      let x = if x > c3 then c3
      let row = update row p x
      let p = p + 1
    else
      let p = 1
      let char2p = 0
      let D = i
      let x = i
    if i <= half + 1
      let e = m + i - half - 2

    loop while p <= e do
      let D = D - 1
      let c3 = D + u16.bool (char1 != b[char2p])
      let char2p = char2p + 1
      let x = x + 1
      let x = if x > c3 then x = c3
      let D = row[p] + 1
      let x = if x > D then x = D
      let row = update row p x
      let p = p + 1

    if i <= half then
      let D = D - 1
      let c3 = D + u16.bool (char1 != b[char2p])
      let x = x + 1
      let x = if x > c3 then x = c3
      let row = update row p x

  -- return row[e]
