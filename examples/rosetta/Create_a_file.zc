import "std/fs.zc"

fn try_create_file(path: char*) {
    println "Creating file: {path}";
    let res = File::open(path, "w");
    if res.is_ok() {
        println "Success: {path} created.";
        res.unwrap().close();
    } else {
        println "Error: {res.err}";
    }
}

fn try_create_dir(path: char*) {
    println "Creating directory: {path}";
    let res = File::create_dir(path);
    if res.is_ok() {
        println "Success: {path} created.";
    } else {
        println "Error: {res.err}";
    }
}

fn main() {
    // Current directory
    try_create_file("output.txt");
    try_create_dir("docs");

    // Filesystem root
    try_create_file("/output.txt");
    try_create_dir("/docs");
}
