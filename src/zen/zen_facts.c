// SPDX-License-Identifier: MIT

#include "zen_facts.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "../parser/parser.h"

typedef struct
{
    ZenTrigger trigger;
    const char *message;
    const char *url;
} ZenFact;

#include "cJSON.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

static ZenFact *facts = NULL;
static int fact_count = 0;
static int has_triggered = 0;

static ZenTrigger map_trigger_name(const char *name)
{
    if (strcmp(name, "TRIGGER_GOTO") == 0)
    {
        return TRIGGER_GOTO;
    }
    if (strcmp(name, "TRIGGER_POINTER_ARITH") == 0)
    {
        return TRIGGER_POINTER_ARITH;
    }
    if (strcmp(name, "TRIGGER_BITWISE") == 0)
    {
        return TRIGGER_BITWISE;
    }
    if (strcmp(name, "TRIGGER_RECURSION") == 0)
    {
        return TRIGGER_RECURSION;
    }
    if (strcmp(name, "TRIGGER_TERNARY") == 0)
    {
        return TRIGGER_TERNARY;
    }
    if (strcmp(name, "TRIGGER_ASM") == 0)
    {
        return TRIGGER_ASM;
    }
    if (strcmp(name, "TRIGGER_WHILE_TRUE") == 0)
    {
        return TRIGGER_WHILE_TRUE;
    }
    if (strcmp(name, "TRIGGER_MACRO") == 0)
    {
        return TRIGGER_MACRO;
    }
    if (strcmp(name, "TRIGGER_VOID_PTR") == 0)
    {
        return TRIGGER_VOID_PTR;
    }
    if (strcmp(name, "TRIGGER_MAIN") == 0)
    {
        return TRIGGER_MAIN;
    }
    if (strcmp(name, "TRIGGER_FORMAT_STRING") == 0)
    {
        return TRIGGER_FORMAT_STRING;
    }
    if (strcmp(name, "TRIGGER_STRUCT_PADDING") == 0)
    {
        return TRIGGER_STRUCT_PADDING;
    }
    if (strcmp(name, "TRIGGER_GLOBAL") == 0)
    {
        return TRIGGER_GLOBAL;
    }
    return TRIGGER_GLOBAL; // Default
}

static void load_facts(void)
{
    const char *search_paths[] = {"src/zen/facts.json", // Dev path
                                  "facts.json",         // CWD
#ifdef ZEN_SHARE_DIR
                                  ZEN_SHARE_DIR "/facts.json", // Install path
#endif
                                  "/usr/local/share/zenc/facts.json",
                                  "/usr/share/zenc/facts.json",
                                  NULL};

    FILE *f = NULL;
    for (int i = 0; search_paths[i]; i++)
    {
        f = fopen(search_paths[i], "r");
        if (f)
        {
            break;
        }
    }

    if (!f)
    {
        return;
    }

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (len < 0)
    {
        fclose(f);
        return;
    }

    char *data = malloc(len + 1);
    if (data)
    {
        size_t read_bytes = fread(data, 1, len, f);
        data[read_bytes] = 0;
    }
    fclose(f);

    if (!data)
    {
        return;
    }

    cJSON *json = cJSON_Parse(data);
    zfree(data);

    if (!json)
    {
        return;
    }

    if (cJSON_IsArray(json))
    {
        fact_count = cJSON_GetArraySize(json);
        if (fact_count > 0)
        {
            facts = calloc(fact_count, sizeof(ZenFact));
        }
        if (!facts && fact_count > 0)
        {
            cJSON_Delete(json);
            return;
        }

        cJSON *item = NULL;
        int i = 0;
        cJSON_ArrayForEach(item, json)
        {
            cJSON *trig = cJSON_GetObjectItem(item, "trigger");
            cJSON *msg = cJSON_GetObjectItem(item, "message");
            cJSON *url = cJSON_GetObjectItem(item, "url");

            if (cJSON_IsString(trig))
            {
                facts[i].trigger = map_trigger_name(trig->valuestring);
            }
            if (cJSON_IsString(msg))
            {
                facts[i].message = strdup(msg->valuestring);
            }
            if (cJSON_IsString(url))
            {
                facts[i].url = strdup(url->valuestring);
            }

            i++;
        }
    }
    cJSON_Delete(json);
}

void zen_init(void)
{
    // Seed random with current time
    srand((unsigned int)(((uint64_t)(z_get_time() * 1000.0)) & 0xFFFFFFFF) ^ z_get_pid());
}

// Global helper to print.
void zzen_at(Token t, const char *msg, const char *url)
{
    fprintf(stderr, COLOR_GREEN "zen: " COLOR_RESET COLOR_BOLD "%s" COLOR_RESET "\n", msg);

    if (t.line > 0)
    {
        const char *zf = g_parser_ctx ? g_parser_ctx->current_filename : "unknown";
        fprintf(stderr, COLOR_BLUE "  --> " COLOR_RESET "%s:%d:%d\n", zf, t.line, t.col);
    }

    if (t.start && t.col > 0)
    {
        const char *line_start = t.start - (t.col - 1);
        const char *line_end = t.start;
        while (*line_end && '\n' != *line_end)
        {
            line_end++;
        }
        int line_len = (int)(line_end - line_start);

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

int zen_trigger_at(ZenTrigger t, Token location, CompilerConfig *cfg)
{
    if (cfg->quiet || !cfg->zen_mode)
    {
        return 0;
    }

    if (has_triggered)
    {
        return 0;
    }

    // g_warning_count is available via zprep.h
    if (g_warning_count > 0)
    {
        return 0;
    }

    if (!facts)
    {
        load_facts();
        if (!facts)
        {
            return 0;
        }
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

void zen_trigger_global(CompilerConfig *cfg)
{
    if (cfg->quiet || !cfg->zen_mode)
    {
        return;
    }
    if (!z_isatty(fileno(stderr)))
    {
        return;
    }
    if (has_triggered)
    {
        return;
    }

    // g_warning_count is available via zprep.h
    if (g_warning_count > 0)
    {
        return;
    }

    if (!facts)
    {
        load_facts();
        if (!facts)
        {
            return;
        }
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
