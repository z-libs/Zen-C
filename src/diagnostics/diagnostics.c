#include "diagnostics.h"
#include "constants.h"
#include "parser.h"
#include "lsp/cJSON.h"
#include <stdio.h>

static CompilerConfig *diag_cfg(void)
{
    return g_parser_ctx ? g_parser_ctx->config : &g_compiler.config;
}

static void emit_json(const char *level, Token t, const char *msg, const char *suggestion,
                      int diag_id)
{
    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(
        root, "file", t.file ? t.file : (g_current_filename ? g_current_filename : "unknown"));
    cJSON_AddNumberToObject(root, "line", t.line);
    cJSON_AddNumberToObject(root, "col", t.col);
    cJSON_AddStringToObject(root, "level", level);
    cJSON_AddStringToObject(root, "message", msg);
    if (diag_id != DIAG_NONE)
    {
        cJSON_AddNumberToObject(root, "code", diag_id);
    }
    if (suggestion)
    {
        cJSON_AddStringToObject(root, "suggestion", suggestion);
    }

    char *json = cJSON_PrintUnformatted(root);
    if (!diag_cfg()->mode_lsp)
    {
        fprintf(stderr, "%s\n", json);
    }
    zfree(json);
    cJSON_Delete(root);

    // Call LSP diagnostics hook
    if (g_parser_ctx && g_parser_ctx->is_fault_tolerant && g_parser_ctx->on_diagnostic)
    {
        int severity = 1; // Error
        if (strcmp(level, "warning") == 0)
        {
            severity = 2; // Warning
        }
        else if (strcmp(level, "info") == 0)
        {
            severity = 3; // Info
        }

        char full_msg[MAX_ERROR_MSG_LEN * 2];
        if (suggestion)
        {
            snprintf(full_msg, sizeof(full_msg), "%s (Suggestion: %s)", msg, suggestion);
        }
        else
        {
            snprintf(full_msg, sizeof(full_msg), "%s", msg);
        }
        g_parser_ctx->on_diagnostic(g_parser_ctx->error_callback_data, t, severity, full_msg,
                                    diag_id);
    }
}

void zpanic(const char *fmt, ...)
{
    if (diag_cfg()->json_output)
    {
        char msg[MAX_ERROR_MSG_LEN];
        va_list a;
        va_start(a, fmt);
        vsnprintf(msg, sizeof(msg), fmt, a);
        va_end(a);
        emit_json("error", (Token){0}, msg, NULL, DIAG_NONE);
        g_error_count++;
        if (diag_cfg()->mode_lsp)
        {
            return;
        }
        exit(1);
    }
    va_list a;
    va_start(a, fmt);
    fprintf(stderr, COLOR_RED "error: " COLOR_RESET COLOR_BOLD);
    vfprintf(stderr, fmt, a);
    fprintf(stderr, COLOR_RESET "\n");
    va_end(a);
    g_error_count++;
    if (diag_cfg()->mode_lsp)
    {
        return;
    }
    exit(1);
}

void zfatal(const char *fmt, ...)
{
    va_list a;
    va_start(a, fmt);
    fprintf(stderr, "Fatal: ");
    vfprintf(stderr, fmt, a);
    fprintf(stderr, "\n");
    va_end(a);
    if (diag_cfg()->mode_lsp)
    {
        return;
    }
    exit(1);
}

// Warning system (non-fatal).
void zwarn(const char *fmt, ...)
{
    if (diag_cfg()->quiet)
    {
        return;
    }
    if (diag_cfg()->json_output)
    {
        char msg[MAX_ERROR_MSG_LEN];
        va_list a;
        va_start(a, fmt);
        vsnprintf(msg, sizeof(msg), fmt, a);
        va_end(a);
        emit_json("warning", (Token){0}, msg, NULL, DIAG_NONE);
        return;
    }
    g_warning_count++;
    va_list a;
    va_start(a, fmt);
    fprintf(stderr, COLOR_YELLOW "warning: " COLOR_RESET COLOR_BOLD);
    vfprintf(stderr, fmt, a);
    fprintf(stderr, COLOR_RESET "\n");
    va_end(a);
}

void zwarn_at_diag(int diag_id, Token t, const char *fmt, ...)
{
    if (diag_cfg()->quiet)
    {
        return;
    }
    char msg[MAX_ERROR_MSG_LEN];
    va_list a;
    va_start(a, fmt);
    vsnprintf(msg, sizeof(msg), fmt, a);
    va_end(a);

    if (diag_cfg()->json_output)
    {
        emit_json("warning", t, msg, NULL, diag_id);
    }
    else
    {
        // Header: 'warning: message'.
        fprintf(stderr, COLOR_YELLOW "warning: " COLOR_RESET COLOR_BOLD);
        fprintf(stderr, "%s", msg);
        fprintf(stderr, COLOR_RESET "\n");

        // Location.
        fprintf(stderr, COLOR_BLUE "  --> " COLOR_RESET "%s:%d:%d\n",
                (t.file ? t.file : g_current_filename), t.line, t.col);
    }
}

void zwarn_at(Token t, const char *fmt, ...)
{
    if (diag_cfg()->quiet)
    {
        return;
    }
    if (diag_cfg()->json_output)
    {
        char msg[MAX_ERROR_MSG_LEN];
        va_list a;
        va_start(a, fmt);
        vsnprintf(msg, sizeof(msg), fmt, a);
        va_end(a);
        emit_json("warning", t, msg, NULL, DIAG_NONE);
        return;
    }
    // Header: 'warning: message'.
    g_warning_count++;
    va_list a;
    va_start(a, fmt);
    fprintf(stderr, COLOR_YELLOW "warning: " COLOR_RESET COLOR_BOLD);
    vfprintf(stderr, fmt, a);
    fprintf(stderr, COLOR_RESET "\n");
    va_end(a);

    // Location.
    fprintf(stderr, COLOR_BLUE "  --> " COLOR_RESET "%s:%d:%d\n",
            (t.file ? t.file : g_current_filename), t.line, t.col);

    // Context. Only if token has valid data.
    if (t.start)
    {
        const char *line_start = t.start - (t.col - 1);
        const char *line_end = t.start;
        while (*line_end && *line_end != '\n')
        {
            line_end++;
        }
        int line_len = line_end - line_start;

        fprintf(stderr, COLOR_BLUE "   |\n" COLOR_RESET);
        fprintf(stderr, COLOR_BLUE "%-3d| " COLOR_RESET "%.*s\n", t.line, line_len, line_start);
        fprintf(stderr, COLOR_BLUE "   | " COLOR_RESET);

        // Caret.
        for (int i = 0; i < t.col - 1; i++)
        {
            fprintf(stderr, " ");
        }
        fprintf(stderr, COLOR_YELLOW "^ here" COLOR_RESET "\n");
        fprintf(stderr, COLOR_BLUE "   |\n" COLOR_RESET);
        fprintf(stderr, COLOR_BLUE "   |\n" COLOR_RESET);
    }
}

void zwarn_with_suggestion_diag(int diag_id, Token t, const char *msg, const char *suggestion)
{
    if (diag_cfg()->quiet)
    {
        return;
    }
    if (diag_cfg()->json_output)
    {
        emit_json("warning", t, msg, suggestion, diag_id);
        return;
    }
    zwarn_with_suggestion(t, msg, suggestion);
}

void zwarn_with_suggestion(Token t, const char *msg, const char *suggestion)
{
    if (diag_cfg()->quiet)
    {
        return;
    }
    if (diag_cfg()->json_output)
    {
        emit_json("warning", t, msg, suggestion, DIAG_NONE);
        return;
    }

    // Header: 'warning: message'.
    g_warning_count++;
    fprintf(stderr, COLOR_YELLOW "warning: " COLOR_RESET COLOR_BOLD "%s" COLOR_RESET "\n", msg);

    // Location.
    fprintf(stderr, COLOR_BLUE "  --> " COLOR_RESET "%s:%d:%d\n",
            (t.file ? t.file : g_current_filename), t.line, t.col);

    // Context.
    if (t.start)
    {
        const char *line_start = t.start - (t.col - 1);
        const char *line_end = t.start;
        while (*line_end && *line_end != '\n')
        {
            line_end++;
        }
        int line_len = line_end - line_start;

        fprintf(stderr, COLOR_BLUE "   |\n" COLOR_RESET);
        fprintf(stderr, COLOR_BLUE "%-3d| " COLOR_RESET "%.*s\n", t.line, line_len, line_start);
        fprintf(stderr, COLOR_BLUE "   | " COLOR_RESET);

        // Caret.
        for (int i = 0; i < t.col - 1; i++)
        {
            fprintf(stderr, " ");
        }
        fprintf(stderr, COLOR_YELLOW "^ here" COLOR_RESET "\n");
        // Suggestion.
        if (suggestion)
        {
            fprintf(stderr, COLOR_BLUE "   |\n" COLOR_RESET);
            fprintf(stderr, COLOR_CYAN "   = note: " COLOR_RESET "%s\n", suggestion);
        }
    }
}

void zpanic_at(Token t, const char *fmt, ...)
{
    if (diag_cfg()->json_output)
    {
        char msg[MAX_ERROR_MSG_LEN];
        va_list a;
        va_start(a, fmt);
        vsnprintf(msg, sizeof(msg), fmt, a);
        va_end(a);
        emit_json("error", t, msg, NULL, DIAG_NONE);
        if (g_parser_ctx && g_parser_ctx->is_fault_tolerant && g_parser_ctx->on_error)
        {
            g_parser_ctx->had_error = 1;
            g_parser_ctx->on_error(g_parser_ctx->error_callback_data, t, msg);
            return;
        }
        if (diag_cfg()->mode_lsp)
        {
            return;
        }
        exit(1);
    }
    // Header: 'error: message'.
    va_list a;
    va_start(a, fmt);
    fprintf(stderr, COLOR_RED "error: " COLOR_RESET COLOR_BOLD);
    vfprintf(stderr, fmt, a);
    fprintf(stderr, COLOR_RESET "\n");
    va_end(a);

    // Location: '--> file:line:col'.
    fprintf(stderr, COLOR_BLUE "  --> " COLOR_RESET "%s:%d:%d\n",
            (t.file ? t.file : g_current_filename), t.line, t.col);

    // Context line.
    const char *line_start = t.start - (t.col - 1);
    const char *line_end = t.start;
    while (*line_end && *line_end != '\n')
    {
        line_end++;
    }
    int line_len = line_end - line_start;

    // Visual bar.
    fprintf(stderr, COLOR_BLUE "   |\n" COLOR_RESET);
    fprintf(stderr, COLOR_BLUE "%-3d| " COLOR_RESET "%.*s\n", t.line, line_len, line_start);
    fprintf(stderr, COLOR_BLUE "   | " COLOR_RESET);

    // caret
    for (int i = 0; i < t.col - 1; i++)
    {
        fprintf(stderr, " ");
    }
    fprintf(stderr, COLOR_RED "^ here" COLOR_RESET "\n");
    fprintf(stderr, COLOR_BLUE "   |\n" COLOR_RESET);

    g_error_count++;
    if (g_parser_ctx && g_parser_ctx->is_fault_tolerant && g_parser_ctx->on_error)
    {
        // Construct error message buffer
        char msg[MAX_ERROR_MSG_LEN];
        va_list args2;
        va_start(args2, fmt);
        vsnprintf(msg, sizeof(msg), fmt, args2);
        va_end(args2);

        g_parser_ctx->had_error = 1;
        g_parser_ctx->on_error(g_parser_ctx->error_callback_data, t, msg);
        return; // Recover!
    }

    if (diag_cfg()->mode_lsp)
    {
        return;
    }
    exit(1);
}

// Enhanced error with suggestion.
void zpanic_with_suggestion(Token t, const char *msg, const char *suggestion)
{
    if (diag_cfg()->json_output)
    {
        emit_json("error", t, msg, suggestion, DIAG_NONE);
        if (g_parser_ctx && g_parser_ctx->is_fault_tolerant && g_parser_ctx->on_error)
        {
            char full_msg[MAX_ERROR_MSG_LEN];
            snprintf(full_msg, sizeof(full_msg), "%s (Suggestion: %s)", msg,
                     suggestion ? suggestion : "");
            g_parser_ctx->had_error = 1;
            g_parser_ctx->on_error(g_parser_ctx->error_callback_data, t, full_msg);
            g_error_count++;
            return;
        }
        if (diag_cfg()->mode_lsp)
        {
            return;
        }
        exit(1);
    }
    // Header.
    fprintf(stderr, COLOR_RED "error: " COLOR_RESET COLOR_BOLD "%s" COLOR_RESET "\n", msg);

    // Location.
    fprintf(stderr, COLOR_BLUE "  --> " COLOR_RESET "%s:%d:%d\n",
            (t.file ? t.file : g_current_filename), t.line, t.col);

    // Context.
    const char *line_start = t.start - (t.col - 1);
    const char *line_end = t.start;
    while (*line_end && *line_end != '\n')
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
    fprintf(stderr, COLOR_RED "^ here" COLOR_RESET "\n");
    g_error_count++;

    // Suggestion.
    if (suggestion)
    {
        fprintf(stderr, COLOR_BLUE "   |\n" COLOR_RESET);
        fprintf(stderr, COLOR_CYAN "   = help: " COLOR_RESET "%s\n", suggestion);
    }

    if (g_parser_ctx && g_parser_ctx->is_fault_tolerant && g_parser_ctx->on_error)
    {
        // Construct error message buffer
        char full_msg[MAX_ERROR_MSG_LEN];
        snprintf(full_msg, sizeof(full_msg), "%s (Suggestion: %s)", msg,
                 suggestion ? suggestion : "");
        g_parser_ctx->had_error = 1;
        g_parser_ctx->on_error(g_parser_ctx->error_callback_data, t, full_msg);
        return; // Recover!
    }

    if (diag_cfg()->mode_lsp)
    {
        return;
    }
    exit(1);
}

void zpanic_with_hints(Token t, const char *msg, const char *const *hints)
{
    if (diag_cfg()->json_output)
    {
        char combined_hints[MAX_PATH_LEN] = {0};
        if (hints)
        {
            const char *const *h = hints;
            while (*h)
            {
                if (combined_hints[0])
                {
                    strncat(combined_hints, "\n", sizeof(combined_hints) - 1);
                }
                strncat(combined_hints, *h, sizeof(combined_hints) - strlen(combined_hints) - 1);
                h++;
            }
        }
        emit_json("error", t, msg, combined_hints[0] ? combined_hints : NULL, DIAG_NONE);

        if (g_parser_ctx && g_parser_ctx->is_fault_tolerant && g_parser_ctx->on_error)
        {
            char full_msg[MAX_PATH_LEN * 2];
            snprintf(full_msg, sizeof(full_msg), "%s\n%s", msg, combined_hints);
            g_parser_ctx->had_error = 1;
            g_parser_ctx->on_error(g_parser_ctx->error_callback_data, t, full_msg);
            return;
        }
        if (diag_cfg()->mode_lsp)
        {
            return;
        }
        exit(1);
    }

    // Header.
    fprintf(stderr, COLOR_RED "error: " COLOR_RESET COLOR_BOLD "%s" COLOR_RESET "\n", msg);

    // Location.
    fprintf(stderr, COLOR_BLUE "  --> " COLOR_RESET "%s:%d:%d\n",
            (t.file ? t.file : g_current_filename), t.line, t.col);

    // Context.
    const char *line_start = t.start - (t.col - 1);
    const char *line_end = t.start;
    while (*line_end && *line_end != '\n')
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
    fprintf(stderr, COLOR_RED "^ here" COLOR_RESET "\n");

    // Hints.
    if (hints)
    {
        const char *const *h = hints;
        while (*h)
        {
            fprintf(stderr, COLOR_BLUE "   |\n" COLOR_RESET);
            fprintf(stderr, COLOR_CYAN "   = help: " COLOR_RESET "%s\n", *h);
            h++;
        }
    }

    if (g_parser_ctx && g_parser_ctx->is_fault_tolerant && g_parser_ctx->on_error)
    {
        // Construct error message buffer
        char full_msg[MAX_PATH_LEN * 2];
        char combined_hints[MAX_MANGLED_NAME_LEN * 4] = {0};
        if (hints)
        {
            const char *const *h = hints;
            while (*h)
            {
                strncat(combined_hints,
                        "\nHelp: ", sizeof(combined_hints) - strlen(combined_hints) - 1);
                strncat(combined_hints, *h, sizeof(combined_hints) - strlen(combined_hints) - 1);
                h++;
            }
        }
        // Construct error message buffer
        int header_len = snprintf(full_msg, sizeof(full_msg), "%s", msg);
        if (header_len < (int)sizeof(full_msg))
        {
            strncat(full_msg, combined_hints, sizeof(full_msg) - strlen(full_msg) - 1);
        }
        g_parser_ctx->had_error = 1;
        g_parser_ctx->on_error(g_parser_ctx->error_callback_data, t, full_msg);
        return; // Recover!
    }

    if (diag_cfg()->mode_lsp)
    {
        return;
    }
    exit(1);
}

void zerror_at(Token t, const char *fmt, ...)
{
    if (diag_cfg()->json_output)
    {
        char msg[MAX_ERROR_MSG_LEN];
        va_list a;
        va_start(a, fmt);
        vsnprintf(msg, sizeof(msg), fmt, a);
        va_end(a);
        emit_json("error", t, msg, NULL, DIAG_NONE);
        g_error_count++;
        if (g_parser_ctx && g_parser_ctx->on_error)
        {
            g_parser_ctx->on_error(g_parser_ctx->error_callback_data, t, msg);
        }
        return;
    }
    // Header: 'error: message'.
    va_list a;
    va_start(a, fmt);
    fprintf(stderr, COLOR_RED "error: " COLOR_RESET COLOR_BOLD);
    vfprintf(stderr, fmt, a);
    fprintf(stderr, COLOR_RESET "\n");
    va_end(a);

    // Location: '--> file:line:col'.
    fprintf(stderr, COLOR_BLUE "  --> " COLOR_RESET "%s:%d:%d\n",
            (t.file ? t.file : g_current_filename), t.line, t.col);

    // Context line.
    if (t.start)
    {
        const char *line_start = t.start - (t.col - 1);
        const char *line_end = t.start;
        while (*line_end && *line_end != '\n')
        {
            line_end++;
        }
        int line_len = line_end - line_start;

        // Visual bar.
        fprintf(stderr, COLOR_BLUE "   |\n" COLOR_RESET);
        fprintf(stderr, COLOR_BLUE "%-3d| " COLOR_RESET "%.*s\n", t.line, line_len, line_start);
        fprintf(stderr, COLOR_BLUE "   | " COLOR_RESET);

        // caret
        for (int i = 0; i < t.col - 1; i++)
        {
            fprintf(stderr, " ");
        }
        fprintf(stderr, COLOR_RED "^ here" COLOR_RESET "\n");
        fprintf(stderr, COLOR_BLUE "   |\n" COLOR_RESET);
    }

    if (g_parser_ctx && g_parser_ctx->on_error)
    {
        // Construct error message buffer
        char msg[MAX_ERROR_MSG_LEN];
        va_list args2;
        va_start(args2, fmt);
        vsnprintf(msg, sizeof(msg), fmt, args2);
        va_end(args2);

        g_parser_ctx->on_error(g_parser_ctx->error_callback_data, t, msg);
    }
    g_error_count++;
}

void zerror_with_suggestion(Token t, const char *msg, const char *suggestion)
{
    if (diag_cfg()->json_output)
    {
        g_error_count++;
        emit_json("error", t, msg, suggestion, DIAG_NONE);
        if (g_parser_ctx && g_parser_ctx->on_error)
        {
            char full_msg[MAX_ERROR_MSG_LEN];
            snprintf(full_msg, sizeof(full_msg), "%s (Suggestion: %s)", msg,
                     suggestion ? suggestion : "");
            g_parser_ctx->on_error(g_parser_ctx->error_callback_data, t, full_msg);
        }
        return;
    }
    // Header.
    fprintf(stderr, COLOR_RED "error: " COLOR_RESET COLOR_BOLD "%s" COLOR_RESET "\n", msg);

    // Location.
    fprintf(stderr, COLOR_BLUE "  --> " COLOR_RESET "%s:%d:%d\n",
            (t.file ? t.file : g_current_filename), t.line, t.col);

    // Context.
    if (t.start)
    {
        const char *line_start = t.start - (t.col - 1);
        const char *line_end = t.start;
        while (*line_end && *line_end != '\n')
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
        fprintf(stderr, COLOR_RED "^ here" COLOR_RESET "\n");

        // Suggestion.
        if (suggestion)
        {
            fprintf(stderr, COLOR_BLUE "   |\n" COLOR_RESET);
            fprintf(stderr, COLOR_CYAN "   = help: " COLOR_RESET "%s\n", suggestion);
        }
    }

    {
        // Construct error message buffer
        char full_msg[MAX_ERROR_MSG_LEN];
        snprintf(full_msg, sizeof(full_msg), "%s (Suggestion: %s)", msg,
                 suggestion ? suggestion : "");
        g_parser_ctx->on_error(g_parser_ctx->error_callback_data, t, full_msg);
    }
    g_error_count++;
}

void zerror_with_hints(Token t, const char *msg, const char *const *hints)
{
    char combined_hints[MAX_PATH_LEN] = {0};
    if (hints)
    {
        const char *const *h = hints;
        while (*h)
        {
            if (combined_hints[0])
            {
                strncat(combined_hints, "\n", sizeof(combined_hints) - 1);
            }
            strncat(combined_hints, *h, sizeof(combined_hints) - strlen(combined_hints) - 1);
            h++;
        }
    }

    if (diag_cfg()->json_output)
    {
        emit_json("error", t, msg, combined_hints[0] ? combined_hints : NULL, DIAG_NONE);
        if (g_parser_ctx && g_parser_ctx->on_error)
        {
            char full_msg[MAX_PATH_LEN * 2];
            int header_len = snprintf(full_msg, sizeof(full_msg), "%s\n", msg);
            if (header_len < (int)sizeof(full_msg))
            {
                strncat(full_msg, combined_hints, sizeof(full_msg) - strlen(full_msg) - 1);
            }
            g_parser_ctx->on_error(g_parser_ctx->error_callback_data, t, full_msg);
        }
        g_error_count++;
        return;
    }

    // Header.
    fprintf(stderr, COLOR_RED "error: " COLOR_RESET COLOR_BOLD "%s" COLOR_RESET "\n", msg);

    // Location.
    fprintf(stderr, COLOR_BLUE "  --> " COLOR_RESET "%s:%d:%d\n",
            (t.file ? t.file : g_current_filename), t.line, t.col);

    // Context.
    if (t.start)
    {
        const char *line_start = t.start - (t.col - 1);
        const char *line_end = t.start;
        while (*line_end && *line_end != '\n')
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
        fprintf(stderr, COLOR_RED "^ here" COLOR_RESET "\n");

        // Hints.
        if (hints)
        {
            const char *const *h = hints;
            while (*h)
            {
                fprintf(stderr, COLOR_BLUE "   |\n" COLOR_RESET);
                fprintf(stderr, COLOR_CYAN "   = help: " COLOR_RESET "%s\n", *h);
                h++;
            }
        }
    }

    if (g_parser_ctx && g_parser_ctx->on_error)
    {
        // Construct error message buffer
        char full_msg[MAX_PATH_LEN * 2];
        int header_len = snprintf(full_msg, sizeof(full_msg), "%s", msg);
        if (header_len < (int)sizeof(full_msg))
        {
            strncat(full_msg, "\n", sizeof(full_msg) - strlen(full_msg) - 1);
            strncat(full_msg, combined_hints, sizeof(full_msg) - strlen(full_msg) - 1);
        }
        g_parser_ctx->on_error(g_parser_ctx->error_callback_data, t, full_msg);
    }
    g_error_count++;
}

// Specific error types with helpful messages.
void error_undefined_function(Token t, const char *func_name, const char *suggestion)
{
    char msg[MAX_SHORT_MSG_LEN];
    snprintf(msg, sizeof(msg), "Undefined function '%s'", func_name);

    if (suggestion)
    {
        char help[MAX_MANGLED_NAME_LEN];
        snprintf(help, sizeof(help), "Did you mean '%s'?", suggestion);
        zerror_with_suggestion(t, msg, help);
    }
    else
    {
        zerror_with_suggestion(t, msg, "Check if the function is defined or imported");
    }
}

void error_wrong_arg_count(Token t, const char *func_name, int expected, int got)
{
    char msg[MAX_SHORT_MSG_LEN];
    snprintf(msg, sizeof(msg), "Wrong number of arguments to function '%s'", func_name);

    char help[MAX_SHORT_MSG_LEN];
    snprintf(help, sizeof(help), "Expected %d argument%s, but got %d", expected,
             expected == 1 ? "" : "s", got);

    zerror_with_suggestion(t, msg, help);
}

void error_undefined_field(Token t, const char *struct_name, const char *field_name,
                           const char *suggestion)
{
    char msg[MAX_SHORT_MSG_LEN];
    snprintf(msg, sizeof(msg), "Struct '%s' has no field '%s'", struct_name, field_name);

    if (suggestion)
    {
        char help[MAX_SHORT_MSG_LEN];
        snprintf(help, sizeof(help), "Did you mean '%s'?", suggestion);
        zerror_with_suggestion(t, msg, help);
    }
    else
    {
        zerror_with_suggestion(t, msg, "Check the struct definition");
    }
}

void error_type_expected(Token t, const char *expected, const char *got)
{
    char msg[MAX_SHORT_MSG_LEN];
    snprintf(msg, sizeof(msg), "Type mismatch");

    char help[MAX_MANGLED_NAME_LEN];
    snprintf(help, sizeof(help), "Expected type '%s', but found '%s'", expected, got);

    zerror_with_suggestion(t, msg, help);
}

void error_cannot_index(Token t, const char *type_name)
{
    char msg[MAX_SHORT_MSG_LEN];
    snprintf(msg, sizeof(msg), "Cannot index into type '%s'", type_name);

    zerror_with_suggestion(t, msg, "Only arrays and slices can be indexed");
}

void warn_unused_variable(Token t, const char *var_name)
{
    if (!is_diag_enabled(DIAG_UNUSED_VAR))
    {
        return;
    }
    char msg[MAX_SHORT_MSG_LEN];
    snprintf(msg, sizeof(msg), "Unused variable '%s'", var_name);
    zwarn_with_suggestion(t, msg, "Consider removing it or prefixing with '_'");
}

void warn_shadowing(Token t, const char *var_name)
{
    if (!is_diag_enabled(DIAG_STYLE_SHADOWING))
    {
        return;
    }
    char msg[MAX_SHORT_MSG_LEN];
    snprintf(msg, sizeof(msg), "Variable '%s' shadows a previous declaration", var_name);
    zwarn_with_suggestion(t, msg, "This can lead to confusion");
}

void warn_unreachable_code(Token t)
{
    if (!is_diag_enabled(DIAG_LOGIC_UNREACHABLE))
    {
        return;
    }
    zwarn_with_suggestion(t, "Unreachable code detected", "This code will never execute");
}

void warn_implicit_conversion(Token t, const char *from_type, const char *to_type)
{
    if (!is_diag_enabled(DIAG_CONVERSION_IMPLICIT))
    {
        return;
    }
    char msg[MAX_SHORT_MSG_LEN];
    snprintf(msg, sizeof(msg), "Implicit conversion from '%s' to '%s'", from_type, to_type);
    zwarn_with_suggestion(t, msg, "Consider using an explicit cast");
}

void warn_missing_return(Token t, const char *func_name)
{
    if (!is_diag_enabled(DIAG_LOGIC_MISSING_RETURN))
    {
        return;
    }
    char msg[MAX_SHORT_MSG_LEN];
    snprintf(msg, sizeof(msg), "Function '%s' may not return a value in all paths", func_name);
    zwarn_with_suggestion(t, msg, "Add a return statement or make the function return 'void'");
}

void warn_comparison_always_true(Token t, const char *reason)
{
    if (!is_diag_enabled(DIAG_LOGIC_ALWAYS_TRUE))
    {
        return;
    }
    zwarn_with_suggestion(t, "Comparison is always true", reason);
}

void warn_comparison_always_false(Token t, const char *reason)
{
    if (!is_diag_enabled(DIAG_LOGIC_ALWAYS_FALSE))
    {
        return;
    }
    zwarn_with_suggestion(t, "Comparison is always false", reason);
}

void warn_unused_parameter(Token t, const char *param_name, const char *func_name)
{
    if (!is_diag_enabled(DIAG_UNUSED_PARAM))
    {
        return;
    }
    char msg[MAX_SHORT_MSG_LEN];
    snprintf(msg, sizeof(msg), "Unused parameter '%s' in function '%s'", param_name, func_name);
    zwarn_with_suggestion(t, msg, "Consider prefixing with '_' if intentionally unused");
}

void warn_narrowing_conversion(Token t, const char *from_type, const char *to_type)
{
    if (!is_diag_enabled(DIAG_CONVERSION_NARROWING))
    {
        return;
    }
    char msg[MAX_SHORT_MSG_LEN];
    snprintf(msg, sizeof(msg), "Narrowing conversion from '%s' to '%s'", from_type, to_type);
    zwarn_with_suggestion(t, msg, "This may cause data loss");
}

void warn_division_by_zero(Token t)
{
    if (!is_diag_enabled(DIAG_SAFETY_DIV_ZERO))
    {
        return;
    }
    zwarn_with_suggestion(t, "Division by zero", "This will cause undefined behavior at runtime");
}

void warn_integer_overflow(Token t, const char *type_name, long long value)
{
    if (!is_diag_enabled(DIAG_SAFETY_INTEGER_OVERFLOW))
    {
        return;
    }
    char msg[MAX_SHORT_MSG_LEN];
    snprintf(msg, sizeof(msg), "Integer literal %lld overflows type '%s'", value, type_name);
    zwarn_with_suggestion(t, msg, "Value will be truncated");
}

void warn_array_bounds(Token t, int index, int size)
{
    if (!is_diag_enabled(DIAG_SAFETY_ARRAY_BOUNDS))
    {
        return;
    }
    char msg[MAX_SHORT_MSG_LEN];
    snprintf(msg, sizeof(msg), "Array index %d is out of bounds for array of size %d", index, size);
    char note[MAX_SHORT_MSG_LEN];
    snprintf(note, sizeof(note), "Valid indices are 0 to %d", size - 1);
    zwarn_with_suggestion(t, msg, note);
}

void warn_format_string(Token t, int arg_num, const char *expected, const char *got)
{
    if (!is_diag_enabled(DIAG_STYLE_FORMAT))
    {
        return;
    }
    char msg[MAX_SHORT_MSG_LEN];
    snprintf(msg, sizeof(msg), "Format argument %d: expected '%s', got '%s'", arg_num, expected,
             got);
    zwarn_with_suggestion(t, msg, "Mismatched format specifier may cause undefined behavior");
}

void warn_null_pointer(Token t, const char *expr)
{
    if (!is_diag_enabled(DIAG_SAFETY_NULL_PTR))
    {
        return;
    }
    char msg[MAX_SHORT_MSG_LEN];
    snprintf(msg, sizeof(msg), "Potential null pointer access in '%s'", expr);
    zwarn_with_suggestion(t, msg, "Add a null check before accessing");
}

void warn_void_main(Token t)
{
    if (!is_diag_enabled(DIAG_PEDANTIC_STRICT_TYPING))
    {
        return;
    }
    zwarn_with_suggestion(t, "'void main()' is non-standard and leads to undefined behavior",
                          "Consider using 'fn main()' or 'fn main() -> c_int' instead");
}

void warn_misra_violation(Token t, const char *msg)
{
    // Fire the MISRA warning if the explicit diag is enabled or if the global compiler flag is
    // active.
    if (!is_diag_enabled(DIAG_MISRA_VIOLATION) && !diag_cfg()->misra_mode)
    {
        return;
    }
    zwarn_with_suggestion(t, msg, "This pattern violates stringent MISRA C safety standards.");
}

int is_diag_enabled(DiagnosticID id)
{
    if (id == DIAG_NONE || id >= DIAG_MAX)
    {
        return 0;
    }
    // Check bitmask in g_config
    return (diag_cfg()->diag_mask & ((uint64_t)1 << id)) != 0;
}

void zwarn_diag(DiagnosticID id, Token t, const char *msg, const char *hint)
{
    if (!is_diag_enabled(id))
    {
        return;
    }

    char final_hint[MAX_MANGLED_NAME_LEN];
    if (id == DIAG_INTEROP_UNDEF_FUNC)
    {
        if (hint)
        {
            // Strip trailing period from hint if it exists to avoid double period
            char hint_copy[MAX_SHORT_MSG_LEN];
            strncpy(hint_copy, hint, sizeof(hint_copy) - 1);
            hint_copy[sizeof(hint_copy) - 1] = '\0';
            size_t len = strlen(hint_copy);
            if (len > 0 && hint_copy[len - 1] == '.')
            {
                hint_copy[len - 1] = '\0';
            }

            snprintf(final_hint, sizeof(final_hint),
                     "%s. If this is a C function, it might need to be whitelisted in 'zenc.json'",
                     hint_copy);
        }
        else
        {
            snprintf(final_hint, sizeof(final_hint),
                     "If this is a C function, it might need to be whitelisted in 'zenc.json'");
        }
        zwarn_with_suggestion_diag(id, t, msg, final_hint);
    }
    else
    {
        zwarn_with_suggestion_diag(id, t, msg, hint);
    }
}

int set_diag_by_name(const char *name, int enabled)
{
    if (strcmp(name, "interop") == 0)
    {
        if (enabled)
        {
            diag_cfg()->diag_mask |= ((uint64_t)1 << DIAG_INTEROP_UNDEF_FUNC);
        }
        else
        {
            diag_cfg()->diag_mask &= ~((uint64_t)1 << DIAG_INTEROP_UNDEF_FUNC);
        }
        return 1;
    }
    else if (strcmp(name, "pedantic") == 0)
    {
        if (enabled)
        {
            diag_cfg()->diag_mask |= ((uint64_t)1 << DIAG_PEDANTIC_STRICT_TYPING);
            diag_cfg()->diag_mask |= ((uint64_t)1 << DIAG_INTEROP_UNDEF_FUNC);
        }
        else
        {
            diag_cfg()->diag_mask &= ~((uint64_t)1 << DIAG_PEDANTIC_STRICT_TYPING);
            diag_cfg()->diag_mask &= ~((uint64_t)1 << DIAG_INTEROP_UNDEF_FUNC);
        }
        return 1;
    }
    else if (strcmp(name, "unused") == 0)
    {
        if (enabled)
        {
            diag_cfg()->diag_mask |= ((uint64_t)1 << DIAG_UNUSED_VAR);
            diag_cfg()->diag_mask |= ((uint64_t)1 << DIAG_UNUSED_PARAM);
        }
        else
        {
            diag_cfg()->diag_mask &= ~((uint64_t)1 << DIAG_UNUSED_VAR);
            diag_cfg()->diag_mask &= ~((uint64_t)1 << DIAG_UNUSED_PARAM);
        }
        return 1;
    }
    else if (strcmp(name, "safety") == 0)
    {
        if (enabled)
        {
            diag_cfg()->diag_mask |= ((uint64_t)1 << DIAG_SAFETY_NULL_PTR);
            diag_cfg()->diag_mask |= ((uint64_t)1 << DIAG_SAFETY_DIV_ZERO);
            diag_cfg()->diag_mask |= ((uint64_t)1 << DIAG_SAFETY_ARRAY_BOUNDS);
        }
        else
        {
            diag_cfg()->diag_mask &= ~((uint64_t)1 << DIAG_SAFETY_NULL_PTR);
            diag_cfg()->diag_mask &= ~((uint64_t)1 << DIAG_SAFETY_DIV_ZERO);
            diag_cfg()->diag_mask &= ~((uint64_t)1 << DIAG_SAFETY_ARRAY_BOUNDS);
        }
        return 1;
    }
    else if (strcmp(name, "logic") == 0)
    {
        if (enabled)
        {
            diag_cfg()->diag_mask |= ((uint64_t)1 << DIAG_LOGIC_UNREACHABLE);
            diag_cfg()->diag_mask |= ((uint64_t)1 << DIAG_LOGIC_ALWAYS_TRUE);
            diag_cfg()->diag_mask |= ((uint64_t)1 << DIAG_LOGIC_ALWAYS_FALSE);
        }
        else
        {
            diag_cfg()->diag_mask &= ~((uint64_t)1 << DIAG_LOGIC_UNREACHABLE);
            diag_cfg()->diag_mask &= ~((uint64_t)1 << DIAG_LOGIC_ALWAYS_TRUE);
            diag_cfg()->diag_mask &= ~((uint64_t)1 << DIAG_LOGIC_ALWAYS_FALSE);
        }
        return 1;
    }
    else if (strcmp(name, "conversion") == 0)
    {
        if (enabled)
        {
            diag_cfg()->diag_mask |= ((uint64_t)1 << DIAG_CONVERSION_IMPLICIT);
            diag_cfg()->diag_mask |= ((uint64_t)1 << DIAG_CONVERSION_NARROWING);
        }
        else
        {
            diag_cfg()->diag_mask &= ~((uint64_t)1 << DIAG_CONVERSION_IMPLICIT);
            diag_cfg()->diag_mask &= ~((uint64_t)1 << DIAG_CONVERSION_NARROWING);
        }
        return 1;
    }
    else if (strcmp(name, "style") == 0)
    {
        if (enabled)
        {
            diag_cfg()->diag_mask |= ((uint64_t)1 << DIAG_STYLE_SHADOWING);
            diag_cfg()->diag_mask |= ((uint64_t)1 << DIAG_STYLE_FORMAT);
        }
        else
        {
            diag_cfg()->diag_mask &= ~((uint64_t)1 << DIAG_STYLE_SHADOWING);
            diag_cfg()->diag_mask &= ~((uint64_t)1 << DIAG_STYLE_FORMAT);
        }
        return 1;
    }
    else if (strcmp(name, "all") == 0)
    {
        if (enabled)
        {
            diag_cfg()->diag_mask = 0xFFFFFFFFFFFFFFFF;
        }
        else
        {
            diag_cfg()->diag_mask = 0;
        }
        return 1;
    }
    return 0;
}
