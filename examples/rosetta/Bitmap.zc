import "std/mem.zc"
import "std/core.zc"

struct Color {
    r: u8;
    g: u8;
    b: u8;
}

struct Image {
    width: usize;
    height: usize;
    pixels: Color*;
}

impl Image {
    fn new(width: usize, height: usize) -> Self {
        return Self {
            width: width,
            height: height,
            pixels: alloc_n<Color>(width * height)
        };
    }

    fn fill(self, color: Color) {
        for let i: usize = 0; i < self.width * self.height; i++ {
            self.pixels[i] = color;
        }
    }

    fn set_pixel(self, x: usize, y: usize, color: Color) {
        if x < self.width && y < self.height {
            self.pixels[y * self.width + x] = color;
        }
    }

    fn get_pixel(self, x: usize, y: usize) -> Color {
        if x < self.width && y < self.height {
            return self.pixels[y * self.width + x];
        }
        return Color { r: 0, g: 0, b: 0 };
    }
}

// RAII: Automatically free memory when the struct goes out of scope
impl Drop for Image {
    fn drop(self) {
        if self.pixels != NULL {
            println "[RAII] Freeing image buffer ({self.width}x{self.height})";
            free(self.pixels);
        }
    }
}

fn main() {
    let img = Image::new(3, 3);
    println "Created {img.width}x{img.height} image.";

    // Fill image with blue
    let blue = Color { r: 0, g: 0, b: 255 };
    img.fill(blue);
    println "Image filled with blue (0, 0, 255).";

    // Set center pixel to red
    let red = Color { r: 255, g: 0, b: 0 };
    img.set_pixel(1, 1, red);
    println "Set pixel at (1, 1) to red (255, 0, 0).";

    // Get colors and verify
    let c11 = img.get_pixel(1, 1);
    let c00 = img.get_pixel(0, 0);

    println "Pixel (1, 1): R={c11.r}, G={c11.g}, B={c11.b}";
    println "Pixel (0, 0): R={c00.r}, G={c00.g}, B={c00.b}";

    assert(c11.r == 255 && c11.g == 0 && c11.b == 0, "Pixel (1,1) should be red");
    assert(c00.r == 0 && c00.g == 0 && c00.b == 255, "Pixel (0,0) should be blue");

    println "Verification successful!";
}
