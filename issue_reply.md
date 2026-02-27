# GitHub Issue Reply: 2D Array Parameters

Thanks for reporting this! You've actually uncovered a bug *and* touched on a fundamental difference in how Zen C represents fixed-size arrays versus dynamic slices in memory.

### 1. The Bug (Fixed in `main`)
The error you saw (`out.c:105:29: error: unknown type name ‘Slice_double’`) was indeed a compiler bug. When parsing the `[[f64]]` syntax, Zen C was generating the C `typedef`s in the wrong order (`Slice_Slice_double` before `Slice_double`). This has just been fixed!

### 2. The Fundamental Mismatch (`[[f64]]` vs `f64[2][2]`)
Even with that bug fixed, your original code (`fn twod(a: [[f64]])` taking `b: f64[2][2]`) will fail to compile. This is because these two types have completely distinct memory layouts:

- `f64[2][2]` compiles into a contiguous 16-byte C array (`double b[2][2]`).
- `[[f64]]` means `Slice<Slice<f64>>`. A `Slice` in Zen C is a fat pointer struct containing `{ data*, int len, int cap }`. So a `Slice<Slice<f64>>` expects an array of these *structs*, which breaks the contiguous layout.

Because of this, you cannot implicitly pass a flat 2D fixed-size array to a dynamically sized slice of slices.

### The Working Solution
Zen C fully supports fixed-size N-dimensional array parameters. Since your array literals are contiguous, you simply need to declare the function argument with the exact contiguous type `f64[2][2]`. Then, Zen C transparently handles bounds-checking.

Here is your working example:

```zc
fn twod(a: f64[2][2]) {
    for i in 0 .. 2 {
        for j in 0 .. 2 {
            println "{a[i][j]}"
        }
    }
}

fn main() {
    // b is a contiguous double[2][2]
    let b: f64[2][2] = [[1.0, 2.0], [3.0, 4.0]];
    twod(b);
}
```

If you truly need dynamically sized multi-dimensional arrays, you would need to use flat 1-D arrays with index math (`idx = y * width + x`), or explicitly construct vectors of vectors `Vec<Vec<f64>>`. But for fixed sizes, the `f64[N][M]` syntax maps directly to C and works perfectly!
