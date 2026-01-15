
#include "zprep_plugin.h"

void bf_transpile(const char *input_body, const ZApi *api)
{
    FILE *out = api->out;
    fprintf(out, "{\n    static unsigned char tape[30000] = {0};\n    unsigned "
                 "char *ptr = tape;\n");
    const char *c = input_body;
    while (*c)
    {
        switch (*c)
        {
        case '>':
            fprintf(out, "    ++ptr;\n");
            break;
        case '<':
            fprintf(out, "    --ptr;\n");
            break;
        case '+':
            fprintf(out, "    ++*ptr;\n");
            break;
        case '-':
            fprintf(out, "    --*ptr;\n");
            break;
        case '.':
            fprintf(out, "    putchar(*ptr);\n");
            break;
        case ',':
            fprintf(out, "    *ptr = getchar();\n");
            break;
        case '[':
            fprintf(out, "    while (*ptr) {\n");
            break;
        case ']':
            fprintf(out, "    }\n");
            break;
        }
        c++;
    }
    fprintf(out, "}\n");
}

ZPlugin brainfuck_plugin = {.name = "brainfuck", .fn = bf_transpile};

PLUGINAPI ZPlugin *z_plugin_init(void)
{
    return &brainfuck_plugin;
}
