# Standard Library: Sort (`std/sort.zc`)

The `std/sort` module provides highly optimized sorting algorithms. It natively implements the `QuickSort` algorithm using a zero-overhead polymorphic macro engine.

## Usage

```zc
import "std/sort.zc"

fn main() {
    let arr: int[5] = [52, 13, 99, 4, 42];
    sort_int((int*)arr, 5); // Becomes [4, 13, 42, 52, 99]
}
```

## Functions

| Method | Signature | Description |
| :--- | :--- | :--- |
| **sort_int** | `sort_int(arr: int*, len: usize)` | Sorts an array of standard integers `[i32]`. |
| **sort_long** | `sort_long(arr: long*, len: usize)` | Sorts an array of long integers `[i64]`. |
| **sort_float** | `sort_float(arr: float*, len: usize)` | Sorts an array of floats `[f32]`. |
| **sort_double** | `sort_double(arr: double*, len: usize)` | Sorts an array of doubles `[f64]`. |

## Custom Sorting

If you create a custom `struct` with `<` operator overloading, you can generate a custom sorter:

```zc
// Emits `sort_MyStruct(MyStruct* arr, usize len)`
raw { ZC_IMPL_SORT(MyStruct) } 
```
