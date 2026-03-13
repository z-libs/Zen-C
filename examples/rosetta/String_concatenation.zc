import "std/string.zc"

fn main() {
    let s1 = String::from("Hello");

    // Instead of raw pointers, wrap the literal in a String and utilize native operator overloading (+).
    // Note: The right operand for overloaded operators generally needs an address reference '&'.
    // We pass by reference ('&') so the operator only *borrows* the string.
    // If we passed by value, the operator function would take ownership and free it.
    let part = String::from(" World");
    let s2 = s1 + &part;

    // We can also use += assignments...
    let excited = String::from("!!");
    s2 += &excited;

    // If we don't want to create intermediate String wrappers, we can use the 'append_c' method
    // on a string object to push raw C-strings directly.
    s2.append_c(" Welcome ");

    // Alternatively, 'append' accepts a pointer to another String object, matching '+='.
    let to_zen = String::from("to Zen C!");
    s2.append(&to_zen);

    println "Original string: {s1}";
    println "Concatenated string: {s2}";
}
