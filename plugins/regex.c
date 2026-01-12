
#include "zprep_plugin.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void emit_match_logic(const char *pattern, FILE *out, int *label_counter);

void regex_transpile(const char *input_body, const ZApi *api)
{
    FILE *out = api->out;
    const char *p = input_body;
    static int regex_counter = 0;

    // Trim whitespace from start/end of pattern block
    while (*p && isspace(*p))
    {
        p++;
    }
    char *pattern = strdup(p);
    int len = strlen(pattern);
    while (len > 0 && isspace(pattern[len - 1]))
    {
        pattern[--len] = 0;
    }

    // Generate unique function name
    int fn_id = regex_counter++;
    char fn_name[64];
    snprintf(fn_name, sizeof(fn_name), "_regex_match_%d", fn_id);

    FILE *target = api->hoist_out ? api->hoist_out : out;

    fprintf(target, "static int %s(const char *text) {\n", fn_name);
    fprintf(target, "    if (!text) return 0;\n");
    fprintf(target, "    const char *c = text;\n");

    int label_id = 0;
    emit_match_logic(pattern, target, &label_id);

    fprintf(target, "    return 1;\n");
    fprintf(target, "}\n");

    fprintf(out, "%s", fn_name);

    free(pattern);
}

static void emit_match_logic(const char *pattern, FILE *out, int *label_counter)
{
    (void)label_counter;
    const char *c = pattern;

    while (*c)
    {
        if (isspace(*c))
        {
            c++;
            continue;
        }

        // Handle Anchors
        if (*c == '^')
        {
            c++;
            continue;
        }

        if (*c == '$')
        {
            fprintf(out, "    if (*c != '\\0') return 0;\n");
            c++;
            continue;
        }

        // Character Class [a-z]
        if (*c == '[')
        {
            const char *scanner = c + 1;
            while (*scanner && *scanner != ']')
            {
                scanner++;
            }
            if (!*scanner)
            {
                return;
            }
            const char *class_end = scanner;

            const char *q = class_end + 1;
            while (*q && isspace(*q))
            {
                q++;
            }
            char quantifier = 0;
            if (*q == '+' || *q == '*')
            {
                quantifier = *q;
            }

            fprintf(out, "    {\n");
            fprintf(out, "      int _count = 0;\n");
            if (quantifier)
            {
                fprintf(out, "      while (1) {\n");
            }

            if (quantifier)
            {
                fprintf(out, "        if (*c == 0) break;\n"); // End of input for loop
            }
            else
            {
                fprintf(out,
                        "        if (*c == 0) return 0;\n"); // End of input for single char
            }

            fprintf(out, "        int match = 0;\n");

            const char *p = c + 1;
            int invert = 0;
            while (p < class_end && isspace(*p))
            {
                p++;
            }
            if (p < class_end && *p == '^')
            {
                invert = 1;
                p++;
            }
            while (p < class_end && isspace(*p))
            {
                p++;
            }

            while (p < class_end)
            {
                if (isspace(*p))
                {
                    p++;
                    continue;
                }

                // Range check
                const char *next = p + 1;
                while (next < class_end && isspace(*next))
                {
                    next++;
                }

                if (next < class_end && *next == '-' && next + 1 < class_end)
                {
                    const char *range_end = next + 1;
                    while (range_end < class_end && isspace(*range_end))
                    {
                        range_end++;
                    }

                    if (range_end < class_end)
                    {
                        fprintf(out, "        if (*c >= '%c' && *c <= '%c') match = 1;\n", *p,
                                *range_end);
                        p = range_end + 1;
                        continue;
                    }
                }

                fprintf(out, "        if (*c == '%c') match = 1;\n", *p);
                p++;
            }

            if (invert)
            {
                fprintf(out, "        if (match) { match = 0; } else { match = 1; }\n");
            }

            if (quantifier)
            {
                fprintf(out, "        if (!match) break;\n");
                fprintf(out, "        c++; _count++;\n");
                fprintf(out, "      }\n"); // End while
                if (quantifier == '+')
                {
                    fprintf(out, "      if (_count == 0) return 0;\n");
                }
            }
            else
            {
                fprintf(out, "      if (!match) return 0;\n");
                fprintf(out, "      c++;\n");
            }

            fprintf(out, "    }\n"); // End Block

            // Advance main pointer c
            c = class_end + 1;
            if (quantifier)
            {
                // Skip the quantifier token we processed
                while (*c && isspace(*c))
                {
                    c++;
                }
                if (*c == quantifier)
                {
                    c++;
                }
            }
            continue;
        }

        // Dot
        if (*c == '.')
        {
            fprintf(out, "    if (*c == 0) return 0; c++;\n");
            c++;
            continue;
        }

        // Literal
        char lit = *c;
        fprintf(out, "    if (*c != '%c') return 0; c++;\n", lit);
        c++;
    }
}

ZPlugin regex_plugin = {.name = "regex", .fn = regex_transpile};
