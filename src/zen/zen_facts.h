
#ifndef ZEN_FACTS_H
#define ZEN_FACTS_H

#include "../zprep.h"

/**
 * @brief Triggers for Zen facts (easter egg system).
 * 
 * Each trigger corresponds to a specific coding pattern or event
 * which may elicit a "Zen Fact" message to the user.
 */
typedef enum
{
    TRIGGER_GOTO,           ///< Usage of `goto`.
    TRIGGER_POINTER_ARITH,  ///< Pointer arithmetic usage.
    TRIGGER_BITWISE,        ///< Bitwise operations.
    TRIGGER_RECURSION,      ///< Recursive calls (currently manual trigger).
    TRIGGER_TERNARY,        ///< Ternary operator usage.
    TRIGGER_ASM,            ///< Inline assembly.
    TRIGGER_WHILE_TRUE,     ///< `while(true)` loops.
    TRIGGER_MACRO,          ///< Macro definitions.
    TRIGGER_VOID_PTR,       ///< `void*` usage.
    TRIGGER_MAIN,           ///< Compilation of `main` function.
    TRIGGER_FORMAT_STRING,  ///< F-string usage.
    TRIGGER_STRUCT_PADDING, ///< Implicit padding detection.
    TRIGGER_GLOBAL          ///< Global variables.
} ZenTrigger;

void zen_init(void);

int zen_trigger_at(ZenTrigger t, Token location);

void zen_trigger_global(void);

const char *zen_get_fact(ZenTrigger t);

#endif
