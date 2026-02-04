
#include "zen_facts.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifndef _WIN32
#include <unistd.h>
#else
#include <io.h>
#include <process.h>
#define STDERR_FILENO 2
#endif
// We keep it low by default.
#define ZEN_PROBABILITY 10

typedef struct
{
    ZenTrigger trigger;
    const char *message;
    const char *url;
} ZenFact;

static const ZenFact facts[] = {
    {TRIGGER_GOTO,
     "Edsger W. Dijkstra considered 'Go To Statement Harmful' in 1968, "
     "advocating for structured "
     "programming.",
     "https://homepages.cwi.nl/~storm/teaching/reader/Dijkstra68.pdf"},
    {TRIGGER_GOTO,
     "Goto can be useful for error cleanup patterns in C (and Zen C), "
     "mimicking 'defer' or "
     "'finally' blocks.",
     NULL},

    {TRIGGER_POINTER_ARITH,
     "In C, `arr[i]` is just `*(arr + i)`. Fun fact: `i[arr]` is also valid "
     "syntax!",
     "https://c-faq.com/aryptr/aryptr2.html"},
    {TRIGGER_POINTER_ARITH,
     "Pointer arithmetic scales by the size of the type. `ptr + 1` increases "
     "the address by "
     "`sizeof(*ptr)` bytes.",
     NULL},

    {TRIGGER_BITWISE, "Use `(x & (x - 1)) == 0` to check if an integer is a power of two.",
     "https://graphics.stanford.edu/~seander/bithacks.html"},
    {TRIGGER_BITWISE,
     "XOR swap algorithm: `x ^= y; y ^= x; x ^= y;` swaps variables without a "
     "temporary (but "
     "optimized code is usually faster).",
     NULL},

    {TRIGGER_RECURSION, "To understand recursion, you must first understand recursion.", NULL},
    {TRIGGER_RECURSION,
     "Tail Call Optimization (TCO) allows some recursive calls to consume no "
     "additional stack "
     "space.",
     "https://en.wikipedia.org/wiki/Tail_call"},

    {TRIGGER_TERNARY,
     "The ternary operator `?:` is the only operator in C that takes three "
     "operands.",
     NULL},

    {TRIGGER_ASM,
     "With great power comes great responsibility. Inline assembly is "
     "compiler-specific and "
     "fragile.",
     NULL},

    {TRIGGER_WHILE_TRUE,
     "The Halting Problem proves it is impossible to determine if an arbitrary "
     "program will "
     "eventually stop.",
     "https://en.wikipedia.org/wiki/Halting_problem"},

    {TRIGGER_MACRO,
     "Macros are handled by the preprocessor, doing simple text replacement "
     "before compilation "
     "begins.",
     NULL},

    {TRIGGER_VOID_PTR,
     "A `void*` is a generic pointer type. You cannot dereference it directly "
     "without casting.",
     NULL},

    {TRIGGER_POINTER_ARITH,
     "Subtracting two pointers returns a `ptrdiff_t`, a signed integer type "
     "capable of holding the "
     "difference.",
     NULL},

    {TRIGGER_MAIN,
     "In C, `main` implicitly returns 0 if no return statement is found "
     "(C99+). Zen C follows "
     "suit.",
     NULL},

    {TRIGGER_FORMAT_STRING,
     "printf format strings are a mini-language interpreted at runtime. Be "
     "careful with user "
     "input!",
     NULL},

    {TRIGGER_BITWISE,
     "The `!!` idiom (double negation) is a standard way to normalize any "
     "non-zero value to "
     "exactly 1.",
     NULL},

    {TRIGGER_GLOBAL,
     "Duff's Device interleaves a switch statement with a do-while loop to "
     "unroll copy loops.",
     "https://en.wikipedia.org/wiki/Duff%27s_device"},
    {TRIGGER_GLOBAL,
     "In C, `sizeof` is an operator, not a function. Parentheses are only "
     "required for type names.",
     NULL},
    {TRIGGER_GLOBAL,
     "Digraphs and Trigraphs were added for keyboards missing symbols. "
     "`\\?\\?=` is `#`, and `<:` "
     "is `[`.",
     "https://en.wikipedia.org/wiki/Digraphs_and_trigraphs"},
    {TRIGGER_GLOBAL,
     "Function designators convert to pointers automatically. `foo`, `&foo`, "
     "and `*foo` all "
     "resolve to the same address.",
     NULL},
    {TRIGGER_GLOBAL,
     "The comma operator evaluates operands left-to-right and returns the "
     "last. `x = (1, 2, 3)` "
     "sets x to 3.",
     NULL},
    {TRIGGER_GLOBAL,
     "Multi-character constants like 'ABCD' are valid C. Their integer value "
     "is "
     "implementation-defined.",
     NULL},
    {TRIGGER_GLOBAL,
     "Bit-fields allow packing struct members into specific bit widths. `int x "
     ": 3;` uses only 3 "
     "bits.",
     NULL},
    {TRIGGER_GLOBAL,
     "Array indexing is commutative: `5[arr]` is semantically identical to "
     "`arr[5]`.",
     NULL},
    {TRIGGER_GLOBAL,
     "The `restrict` keyword promises the compiler that a pointer is the only "
     "access to that "
     "memory.",
     NULL},
    {TRIGGER_GLOBAL,
     "A struct's size can be larger than the sum of its members due to "
     "alignment padding.",
     NULL},
    {TRIGGER_GLOBAL,
     "In C, `void*` arithmetic is a GCC extension (treating size as 1). "
     "Standard C forbids it.",
     NULL},
    {TRIGGER_GLOBAL,
     "The C standard guarantees that `NULL` equals `(void*)0`, but the bit "
     "pattern may not be all "
     "zeros.",
     NULL},
    {TRIGGER_GLOBAL,
     "`sizeof('a')` is 4 in C (int) but 1 in C++ (char). A subtle difference "
     "between the "
     "languages.",
     NULL},
    {TRIGGER_GLOBAL,
     "Static local variables are initialized only once, even if the function "
     "is called multiple "
     "times.",
     NULL},
    {TRIGGER_GLOBAL,
     "The `volatile` keyword prevents compiler optimizations on a variable, "
     "useful for hardware "
     "registers.",
     NULL},
    {TRIGGER_GLOBAL,
     "In C99+, Variable Length Arrays (VLAs) can have runtime-determined "
     "sizes. Use with caution!",
     NULL},
    {TRIGGER_GLOBAL,
     "The `_Alignof` operator (C11) returns the alignment requirement of a "
     "type.",
     NULL},
    {TRIGGER_GLOBAL,
     "Compound literals like `(int[]){1, 2, 3}` create anonymous arrays with "
     "automatic storage.",
     NULL},
    {TRIGGER_GLOBAL,
     "Designated initializers: `.field = val` lets you initialize struct "
     "fields out of order.",
     NULL},
    {TRIGGER_GLOBAL,
     "The `#` stringification operator in macros turns arguments into string "
     "literals.",
     NULL},
    {TRIGGER_GLOBAL, "The `##` token-pasting operator concatenates tokens in macro expansions.",
     NULL},
    {TRIGGER_GLOBAL,
     "Flexible array members: `int data[];` at struct end allows variable-size "
     "structs.",
     NULL},
    {TRIGGER_GLOBAL,
     "Anonymous structs/unions (C11) allow direct member access without field "
     "names.",
     NULL},
    {TRIGGER_GLOBAL,
     "`_Generic` (C11) provides compile-time type dispatch, like a simpler "
     "form of overloading.",
     NULL},
    {TRIGGER_GLOBAL,
     "The `register` keyword is a hint to the compiler, but modern compilers "
     "ignore it.",
     NULL},
    {TRIGGER_GLOBAL,
     "Integer promotion: `char` and `short` are promoted to `int` in "
     "expressions.",
     NULL},
    {TRIGGER_GLOBAL,
     "Signed integer overflow is undefined behavior in C. Use unsigned for "
     "wrap-around arithmetic.",
     NULL},
    {TRIGGER_GLOBAL,
     "The order of evaluation for function arguments is unspecified in C. "
     "Never rely on it!",
     NULL},
    {TRIGGER_GLOBAL,
     "In C, you can take the address of a label with `&&label` (GCC "
     "extension).",
     NULL},
    {TRIGGER_GLOBAL,
     "The `inline` keyword is only a suggestion. Compilers decide whether to "
     "actually inline.",
     NULL},
    {TRIGGER_GLOBAL,
     "`setjmp`/`longjmp` provide non-local jumps, but are dangerous and rarely "
     "needed.",
     NULL},
    {TRIGGER_GLOBAL,
     "The `#pragma once` directive is non-standard but widely supported for "
     "include guards.",
     NULL},
    {TRIGGER_GLOBAL,
     "A `char` can be signed or unsigned depending on the platform. Use "
     "`signed char` or `unsigned "
     "char` to be explicit.",
     NULL},
    {TRIGGER_GLOBAL,
     "The `const` keyword doesn't make data immutable - you can cast it away "
     "(but shouldn't).",
     NULL},
    {TRIGGER_GLOBAL,
     "C has no native boolean type before C99. `_Bool` was added in C99, "
     "`bool` via stdbool.h.",
     NULL},
    {TRIGGER_GLOBAL,
     "Empty parameter lists in C mean 'unspecified arguments', not 'no "
     "arguments'. Use `(void)` "
     "for none.",
     NULL},
    {TRIGGER_GLOBAL,
     "The `extern` keyword declares a variable without defining it, linking to "
     "another translation "
     "unit.",
     NULL},
    {TRIGGER_GLOBAL,
     "K&R style function definitions predate ANSI C and put parameter types "
     "after the parentheses.",
     NULL},
    {TRIGGER_GLOBAL,
     "The IOCCC (Obfuscated C Code Contest) showcases creative abuse of C "
     "syntax since 1984.",
     "https://www.ioccc.org"},
    {TRIGGER_GLOBAL,
     "Dennis Ritchie developed C at Bell Labs between 1969-1973. It replaced "
     "B, which was "
     "typeless.",
     NULL},
    {TRIGGER_GLOBAL, "The name 'C' simply comes from being the successor to the B language.", NULL},
    {TRIGGER_GLOBAL,
     "Plan 9 C allows `structure.member` notation even if `member` is inside "
     "an anonymous inner "
     "struct. C11 finally adopted this!",
     "https://9p.io/sys/doc/comp.html"},
    {TRIGGER_GLOBAL,
     "In Plan 9 C, arrays of zero length `int data[0]` were valid long before "
     "C99 flexible array "
     "members.",
     NULL},
    {TRIGGER_GLOBAL,
     "Plan 9 C headers are 'idempotent' — they contain their own include "
     "guards, so you never see "
     "`#ifndef HEADER_H` boilerplate.",
     NULL},
    {TRIGGER_GLOBAL,
     "In Plan 9, the `nil` pointer is distinct from 0. Accessing `nil` causes "
     "a hardware trap, not "
     "just undefined behavior.",
     NULL},
    {TRIGGER_GLOBAL,
     "Ken Thompson, creator of B and C, also designed UTF-8 encoding on a "
     "placemat in a New Jersey "
     "diner.",
     "https://www.cl.cam.ac.uk/~mgk25/ucs/utf-8-history.txt"},
    {TRIGGER_GLOBAL,
     "C11 introduced `_Noreturn` to tell the compiler a function (like `exit`) "
     "will never return "
     "control to the caller.",
     NULL},
    {TRIGGER_GLOBAL,
     "A 'Sequence Point' is a juncture where all side effects of previous "
     "evaluations must be "
     "complete. Violation = UB.",
     "https://en.wikipedia.org/wiki/Sequence_point"},
    {TRIGGER_GLOBAL,
     "VLAs were mandatory in C99, but made optional in C11 because they are "
     "hard to implement "
     "safely.",
     NULL},
    {TRIGGER_GLOBAL,
     "Pre-ANSI C (K&R) didn't have `void`. Functions returning nothing "
     "actually returned an "
     "undefined `int`.",
     NULL},
    {TRIGGER_GLOBAL,
     "Adjacent string literals are concatenated automatically. `\"Hello \" "
     "\"World\"` becomes "
     "`\"Hello World\"`.",
     NULL},
    {TRIGGER_GLOBAL,
     "The `auto` keyword exists in C! It declares automatic storage duration, "
     "but is almost never "
     "used since it's the default.",
     NULL},
    {TRIGGER_GLOBAL,
     "Bitwise operators have lower precedence than comparisons! `val & MASK == "
     "0` is `val & (MASK "
     "== 0)`. Always parenthesize!",
     NULL},
    {TRIGGER_GLOBAL,
     "The 'as-if' rule allows compilers to transform code however they want, "
     "as long as observable "
     "behavior remains the same.",
     "https://en.cppreference.com/w/c/language/as_if"},
    {TRIGGER_GLOBAL,
     "Tail Recursion Elimination (TRE) isn't guaranteed by the C standard, but "
     "most compilers do "
     "it at -O2.",
     NULL},
    {TRIGGER_GLOBAL,
     "GCC's `__builtin_expect` allows you to tell the branch predictor which "
     "path is more likely "
     "(the `likely()` macro).",
     NULL},
    {TRIGGER_GLOBAL, "The actual inspiration for this project was this video.",
     "https://youtu.be/vXYVfk7agqU?si=fjbB9iwdmL8Qhjol"}, // Have fun.
};

static int fact_count = sizeof(facts) / sizeof(ZenFact);
static int has_triggered = 0;

void zen_init(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    srand(ts.tv_nsec ^ getpid());
}

// Global helper to print.
void zzen_at(Token t, const char *msg, const char *url)
{
    fprintf(stderr, COLOR_GREEN "zen: " COLOR_RESET COLOR_BOLD "%s" COLOR_RESET "\n", msg);

    extern char *g_current_filename;
    if (t.line > 0)
    {
        fprintf(stderr, COLOR_BLUE "  --> " COLOR_RESET "%s:%d:%d\n",
                g_current_filename ? g_current_filename : "unknown", t.line, t.col);
    }

    if (t.start)
    {
        const char *line_start = t.start - (t.col - 1);
        const char *line_end = t.start;
        while (*line_end && '\n' != *line_end)
        {
            line_end++;
        }
        int line_len = line_end - line_start;

        fprintf(stderr, COLOR_BLUE "   |\n" COLOR_RESET);
        fprintf(stderr, COLOR_BLUE "%-3d| " COLOR_RESET "%.*s\n", t.line, line_len, line_start);
        fprintf(stderr, COLOR_BLUE "   | " COLOR_RESET);
        for (int i = 0; i < t.col - 1; i++)
        {
            fprintf(stderr, " ");
        }
        fprintf(stderr, COLOR_GREEN "^ zen tip" COLOR_RESET "\n");
    }

    if (url)
    {
        fprintf(stderr, COLOR_CYAN "   = read more: %s" COLOR_RESET "\n", url);
    }
}

int zen_trigger_at(ZenTrigger t, Token location)
{
    if (g_config.quiet || g_config.no_zen)
    {
        return 0;
    }

    if (has_triggered)
    {
        return 0;
    }

    extern int g_warning_count;
    if (g_warning_count > 0)
    {
        return 0;
    }

    if ((rand() % 100) >= ZEN_PROBABILITY)
    {
        return 0;
    }

    int matches[10];
    int match_count = 0;

    for (int i = 0; i < fact_count; i++)
    {
        if (facts[i].trigger == t)
        {
            matches[match_count++] = i;
            if (match_count >= 10)
            {
                break;
            }
        }
    }

    if (0 == match_count)
    {
        return 0;
    }

    int pick = matches[rand() % match_count];
    const ZenFact *f = &facts[pick];

    zzen_at(location, f->message, f->url);
    has_triggered = 1;
    return 1;
}

void zen_trigger_global(void)
{
    if (g_config.quiet || g_config.no_zen)
    {
        return;
    }
    if (!isatty(STDERR_FILENO))
    {
        return;
    }
    if (has_triggered)
    {
        return;
    }

    extern int g_warning_count;
    if (g_warning_count > 0)
    {
        return;
    }

    if ((rand() % 100) >= ZEN_PROBABILITY)
    {
        return;
    }

    int matches[10];
    int match_count = 0;

    for (int i = 0; i < fact_count; i++)
    {
        if (TRIGGER_GLOBAL == facts[i].trigger)
        {
            matches[match_count++] = i;
            if (match_count >= 10)
            {
                break;
            }
        }
    }

    if (0 == match_count)
    {
        return;
    }

    int pick = matches[rand() % match_count];
    const ZenFact *f = &facts[pick];

    Token empty = {0};
    zzen_at(empty, f->message, f->url);
    has_triggered = 1;
}
