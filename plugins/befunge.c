
#include "zprep_plugin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BF_W 80
#define BF_H 25

void befunge_transpile(const char *input_body, const ZApi *api)
{
    FILE *out = api->out;

    char grid[BF_H][BF_W];
    memset(grid, ' ', sizeof(grid));

    int row = 0, col = 0;
    const char *p = input_body;
    while (*p && (*p == '\n' || *p == '\r'))
    {
        p++;
    }

    while (*p)
    {
        if (*p == '\n')
        {
            row++;
            col = 0;
            if (row >= BF_H)
            {
                break;
            }
        }
        else if (col < BF_W)
        {
            grid[row][col] = *p;
            col++;
        }
        p++;
    }

    fprintf(out, "{\n");
    fprintf(out, "    static long stack[1024]; int sp = 0;\n");
    fprintf(out, "    int x = 0, y = 0, dx = 1, dy = 0;\n");
    fprintf(out, "    int string_mode = 0;\n\n");

    fprintf(out, "    static void *dispatch[%d][%d] = {\n", BF_H, BF_W);
    for (int r = 0; r < BF_H; r++)
    {
        fprintf(out, "        { ");
        for (int c = 0; c < BF_W; c++)
        {
            if (grid[r][c] == ' ')
            {
                fprintf(out, "&&space_handler, ");
            }
            else
            {
                fprintf(out, "&&cell_%d_%d, ", r, c);
            }
        }
        fprintf(out, "},\n");
    }
    fprintf(out, "    };\n\n");
    fprintf(out, "    goto *dispatch[0][0];\n\n");

    for (int r = 0; r < BF_H; r++)
    {
        for (int c = 0; c < BF_W; c++)
        {
            char op = grid[r][c];
            if (op == ' ')
            {
                continue;
            }

            fprintf(out, "cell_%d_%d:\n", r, c);

            if (op >= '0' && op <= '9')
            {
                fprintf(out,
                        "    if(string_mode) { stack[sp++] = '%c'; } else { "
                        "stack[sp++] = %c; }\n",
                        op, op);
            }
            else if (op == '"')
            {
                fprintf(out, "    string_mode = !string_mode;\n");
            }
            else
            {
                if (op == '>')
                {
                    fprintf(out, "    dx=1; dy=0;\n");
                }
                else if (op == '<')
                {
                    fprintf(out, "    dx=-1; dy=0;\n");
                }
                else if (op == '^')
                {
                    fprintf(out, "    dx=0; dy=-1;\n");
                }
                else if (op == 'v')
                {
                    fprintf(out, "    dx=0; dy=1;\n");
                }
                else if (op == '+')
                {
                    fprintf(out, "    { long a=stack[--sp]; stack[sp-1]+=a; }\n");
                }
                else if (op == '-')
                {
                    fprintf(out, "    { long a=stack[--sp]; stack[sp-1]-=a; }\n");
                }
                else if (op == '*')
                {
                    fprintf(out, "    { long a=stack[--sp]; stack[sp-1]*=a; }\n");
                }
                else if (op == '/')
                {
                    fprintf(out, "    { long a=stack[--sp]; stack[sp-1]= "
                                 "(a!=0)?stack[sp-1]/a:0; }\n");
                }
                else if (op == '%')
                {
                    fprintf(out, "    { long a=stack[--sp]; stack[sp-1]= "
                                 "(a!=0)?stack[sp-1]%%a:0; }\n");
                }
                else if (op == '!')
                {
                    fprintf(out, "    stack[sp-1] = !stack[sp-1];\n");
                }
                else if (op == '`')
                {
                    fprintf(out, "    { long a=stack[--sp]; stack[sp-1]=(stack[sp-1]>a); }\n");
                }
                else if (op == ':')
                {
                    fprintf(out, "    if(sp>0) { stack[sp]=stack[sp-1]; sp++; }\n");
                }
                else if (op == '\\')
                {
                    fprintf(out, "   if(sp>1) { long t=stack[sp-1]; stack[sp-1]=stack[sp-2]; "
                                 "stack[sp-2]=t; }\n");
                }
                else if (op == '$')
                {
                    fprintf(out, "    if(sp>0) sp--;\n");
                }
                else if (op == '.')
                {
                    fprintf(out, "    printf(\"%%ld \", stack[--sp]);\n");
                }
                else if (op == ',')
                {
                    fprintf(out, "    printf(\"%%c\", (char)stack[--sp]);\n");
                }
                else if (op == '_')
                {
                    fprintf(out, "    { long a=stack[--sp]; dx=a?-1:1; dy=0; }\n");
                }
                else if (op == '|')
                {
                    fprintf(out, "    { long a=stack[--sp]; dx=0; dy=a?-1:1; }\n");
                }
                else if (op == '@')
                {
                    fprintf(out, "    goto end_befunge;\n");
                }
                else if (op == '#')
                {
                    fprintf(out, "    x+=dx; y+=dy;\n");
                }
                else
                {
                    fprintf(out, "    if(string_mode) stack[sp++] = '%c';\n", op);
                }
            }

            fprintf(out, "    x += dx; y += dy;\n");
            fprintf(out, "    if(x>=%d) x=0; else if(x<0) x=%d-1;\n", BF_W, BF_W);
            fprintf(out, "    if(y>=%d) y=0; else if(y<0) y=%d-1;\n", BF_H, BF_H);
            fprintf(out, "    goto *dispatch[y][x];\n\n");
        }
    }

    fprintf(out, "space_handler:\n");
    fprintf(out, "    x += dx; y += dy;\n");
    fprintf(out, "    if(x>=%d) x=0; else if(x<0) x=%d-1;\n", BF_W, BF_W);
    fprintf(out, "    if(y>=%d) y=0; else if(y<0) y=%d-1;\n", BF_H, BF_H);
    fprintf(out, "    goto *dispatch[y][x];\n\n");

    fprintf(out, "end_befunge:;\n");
    fprintf(out, "}\n");
}

ZPlugin befunge_plugin = {.name = "befunge", .fn = befunge_transpile};

PLUGINAPI ZPlugin *z_plugin_init(void)
{
    return &befunge_plugin;
}
