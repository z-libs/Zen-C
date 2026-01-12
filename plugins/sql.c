
#include "zprep_plugin.h"
#include "compat/compat.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Col
{
    char name[32];
    char type[32];
    struct Col *next;
} Col;

typedef struct Table
{
    char name[64];
    Col *cols;
    struct Table *next;
} Table;

static Table *g_tables = NULL;

static void register_table(const char *name)
{
    Table *t = malloc(sizeof(Table));
    strcpy(t->name, name);
    t->cols = NULL;
    t->next = g_tables;
    g_tables = t;
}

static void add_column(const char *table_name, const char *col_name, const char *col_type)
{
    Table *t = g_tables;
    while (t && strcmp(t->name, table_name) != 0)
    {
        t = t->next;
    }
    if (t)
    {
        Col *c = malloc(sizeof(Col));
        strcpy(c->name, col_name);
        strcpy(c->type, col_type);
        c->next = NULL;

        if (!t->cols)
        {
            t->cols = c;
        }
        else
        {
            Col *last = t->cols;
            while (last->next)
            {
                last = last->next;
            }
            last->next = c;
        }
    }
}

static Table *find_table(const char *name)
{
    Table *t = g_tables;
    while (t)
    {
        if (strcmp(t->name, name) == 0)
        {
            return t;
        }
        t = t->next;
    }
    return NULL;
}

static void skip_whitespace(const char **p)
{
    while (1)
    {
        while (**p && isspace(**p))
        {
            (*p)++;
        }

        if (**p == '/' && *(*p + 1) == '/')
        {
            // Single line comment
            (*p) += 2;
            while (**p && **p != '\n')
            {
                (*p)++;
            }
        }
        else if (**p == '/' && *(*p + 1) == '*')
        {
            // Multi line comment
            (*p) += 2;
            while (**p && !(**p == '*' && *(*p + 1) == '/'))
            {
                (*p)++;
            }
            if (**p)
            {
                (*p) += 2; // Skip */
            }
        }
        else
        {
            break;
        }
    }
}

static int match_kw(const char **p, const char *kw)
{
    size_t len = strlen(kw);
    if (strncasecmp(*p, kw, len) == 0 && !isalnum((*p)[len]) && (*p)[len] != '_')
    {
        *p += len;
        return 1;
    }
    return 0;
}

static void parse_create_table(const char **p, const ZApi *api)
{
    FILE *out = api->out;
    skip_whitespace(p);

    // table_name
    const char *name_start = *p;
    while (**p && (isalnum(**p) || **p == '_'))
    {
        (*p)++;
    }
    int name_len = *p - name_start;
    char table_name[64];
    snprintf(table_name, name_len + 1, "%s", name_start);

    register_table(table_name);

    skip_whitespace(p);
    if (**p == '(')
    {
        (*p)++;
    }
    else
    {
        fprintf(stderr, "Error: Expected '(' after table name\n");
        exit(1);
    }
    skip_whitespace(p);

    fprintf(out, "typedef struct {\n");

    // Columns
    while (**p && **p != ')')
    {
        skip_whitespace(p);
        const char *col_start = *p;
        while (**p && (isalnum(**p) || **p == '_'))
        {
            (*p)++;
        }
        int col_len = *p - col_start;
        if (col_len == 0 && **p != ',')
        {
            // Unexpected token or end
            break;
        }

        char col_name[64];
        snprintf(col_name, col_len + 1, "%s", col_start);

        skip_whitespace(p);
        const char *type_start = *p;
        while (**p && (isalnum(**p) || **p == '_'))
        {
            (*p)++;
        }
        int type_len = *p - type_start;

        char ctype[32] = "int";
        char store_type[32] = "int";

        if (type_len > 0 && (strncasecmp(type_start, "TEXT", type_len) == 0 ||
                             strncasecmp(type_start, "STRING", type_len) == 0))
        {
            strcpy(ctype, "const char*");
            strcpy(store_type, "char*");
        }

        add_column(table_name, col_name, store_type);

        fprintf(out, "    %s %s;\n", ctype, col_name);

        skip_whitespace(p);
        if (**p == ',')
        {
            (*p)++;
        }
    }
    if (**p == ')')
    {
        (*p)++;
    }

    skip_whitespace(p);
    if (**p == ';')
    {
        (*p)++;
    }

    fprintf(out, "} Row_%s;\n", table_name);
    fprintf(out, "static Row_%s table_%s[128];\n", table_name, table_name);
    fprintf(out, "static int count_%s = 0;\n", table_name);
}

static void parse_insert(const char **p, const ZApi *api)
{
    FILE *out = api->out;
    skip_whitespace(p);
    if (!match_kw(p, "INTO"))
    {
        // Well, it is just a toy plugin rn, what did you expect?
    }
    skip_whitespace(p);

    const char *name_start = *p;
    while (**p && (isalnum(**p) || **p == '_'))
    {
        (*p)++;
    }
    int name_len = *p - name_start;
    char table_name[64];
    snprintf(table_name, name_len + 1, "%s", name_start);

    skip_whitespace(p);
    if (!match_kw(p, "VALUES"))
    {
        fprintf(stderr, "Error: Expected VALUES in INSERT\n");
        exit(1);
    }
    skip_whitespace(p);
    if (**p == '(')
    {
        (*p)++;
    }

    fprintf(out, "if (count_%s < 128) {\n", table_name);
    fprintf(out, "    table_%s[count_%s] = (Row_%s){ ", table_name, table_name, table_name);

    while (**p && **p != ')')
    {
        skip_whitespace(p);
        // Value expression
        const char *val_start = *p;
        if (**p == '"')
        {
            (*p)++;
            while (**p && **p != '"')
            {
                if (**p == '\\')
                {
                    (*p)++;
                }
                (*p)++;
            }
            if (**p == '"')
            {
                (*p)++;
            }
        }
        else
        {
            while (**p && **p != ',' && **p != ')')
            {
                (*p)++;
            }
        }
        fwrite(val_start, 1, *p - val_start, out);

        skip_whitespace(p);
        if (**p == ',')
        {
            fprintf(out, ", ");
            (*p)++;
        }
    }
    if (**p == ')')
    {
        (*p)++;
    }
    skip_whitespace(p);
    if (**p == ';')
    {
        (*p)++;
    }

    fprintf(out, " };\n");
    fprintf(out, "    count_%s++;\n", table_name);
    fprintf(out, "}\n");
}

static void parse_select(const char **p, const ZApi *api)
{
    FILE *out = api->out;
    skip_whitespace(p);
    match_kw(p, "*"); // MVP only supports *
    skip_whitespace(p);
    if (!match_kw(p, "FROM"))
    {
        fprintf(stderr, "Error: Expected FROM in SELECT\n");
        exit(1);
    }
    skip_whitespace(p);

    const char *name_start = *p;
    while (**p && (isalnum(**p) || **p == '_'))
    {
        (*p)++;
    }
    int name_len = *p - name_start;
    char table_name[64];
    snprintf(table_name, name_len + 1, "%s", name_start);

    skip_whitespace(p);
    if (**p == ';')
    {
        (*p)++;
    }

    Table *t = find_table(table_name);
    if (!t)
    {
        fprintf(stderr, "Error: Unknown table '%s' in SELECT\n", table_name);
        exit(1);
    }

    fprintf(out, "for(int _i=0; _i<count_%s; _i++) {\n", table_name);
    fprintf(out, "    printf(\"");

    // Format string
    Col *c = t->cols;
    while (c)
    {
        if (strcmp(c->type, "int") == 0)
        {
            fprintf(out, "%%d");
        }
        else
        {
            fprintf(out, "%%s");
        }

        if (c->next)
        {
            fprintf(out, " ");
        }
        c = c->next;
    }
    fprintf(out, "\\n\"");

    // Args
    c = t->cols;
    while (c)
    {
        fprintf(out, ", table_%s[_i].%s", table_name, c->name);
        c = c->next;
    }

    fprintf(out, ");\n");
    fprintf(out, "}\n");
}

void sql_transpile(const char *input_body, const ZApi *api)
{
    FILE *out = api->out;
    const char *p = input_body;

    fprintf(out, "({\n");

    while (*p)
    {
        skip_whitespace(&p);
        if (!*p)
        {
            break;
        }

        if (match_kw(&p, "CREATE"))
        {
            skip_whitespace(&p);
            match_kw(&p, "TABLE"); // consume optional TABLE
            parse_create_table(&p, api);
        }
        else if (match_kw(&p, "INSERT"))
        {
            parse_insert(&p, api);
        }
        else if (match_kw(&p, "SELECT"))
        {
            parse_select(&p, api);
        }
        else
        {
            p++; // Skip unknown or error
        }
    }

    fprintf(out, "0; })"); // Return 0
}

ZPlugin sql_plugin = {.name = "sql", .fn = sql_transpile};
