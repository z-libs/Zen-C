#include "cJSON.h"
#include "lsp_project.h" // Includes lsp_index.h, parser.h
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct Diagnostic
{
    int line;
    int col;
    char *message;
    struct Diagnostic *next;
} Diagnostic;

typedef struct
{
    Diagnostic *head;
    Diagnostic *tail;
} DiagnosticList;

// Helper to send JSON response
static void send_json_response(cJSON *root)
{
    char *str = cJSON_PrintUnformatted(root);
    if (str)
    {
        fprintf(stdout, "Content-Length: %llu\r\n\r\n%s", (unsigned long long)strlen(str), str);
        fflush(stdout);
        free(str);
    }
    cJSON_Delete(root);
}

// Callback for parser errors.
void lsp_on_error(void *data, Token t, const char *msg)
{
    DiagnosticList *list = (DiagnosticList *)data;
    // Simple allocation for MVP
    Diagnostic *d = calloc(1, sizeof(Diagnostic));
    d->line = t.line > 0 ? t.line - 1 : 0;
    d->col = t.col > 0 ? t.col - 1 : 0;
    d->message = strdup(msg);
    d->next = NULL;

    if (!list->head)
    {
        list->head = d;
        list->tail = d;
    }
    else
    {
        list->tail->next = d;
        list->tail = d;
    }
}

void lsp_check_file(const char *uri, const char *json_src, int id)
{
    (void)id;
    if (!g_project)
    {
        char cwd[1024];
        if (getcwd(cwd, sizeof(cwd)))
        {
            lsp_project_init(cwd);
        }
        else
        {
            lsp_project_init(".");
        }
    }

    // Setup error capture on the global project context
    DiagnosticList diagnostics = {0};

    // We attach the callback to 'g_project->ctx'.
    void *old_data = g_project->ctx->error_callback_data;
    void (*old_cb)(void *, Token, const char *) = g_project->ctx->on_error;

    g_project->ctx->error_callback_data = &diagnostics;
    g_project->ctx->on_error = lsp_on_error;

    // Update and Parse
    lsp_project_update_file(uri, json_src);

    // Restore
    g_project->ctx->on_error = old_cb;
    g_project->ctx->error_callback_data = old_data;

    // Construct JSON Response (publishDiagnostics)
    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "jsonrpc", "2.0");
    cJSON_AddStringToObject(root, "method", "textDocument/publishDiagnostics");

    cJSON *params = cJSON_CreateObject();
    cJSON_AddStringToObject(params, "uri", uri);

    cJSON *diag_array = cJSON_CreateArray();

    Diagnostic *d = diagnostics.head;
    while (d)
    {
        cJSON *diag = cJSON_CreateObject();

        cJSON *range = cJSON_CreateObject();
        cJSON *start = cJSON_CreateObject();
        cJSON_AddNumberToObject(start, "line", d->line);
        cJSON_AddNumberToObject(start, "character", d->col);

        cJSON *end = cJSON_CreateObject();
        cJSON_AddNumberToObject(end, "line", d->line);
        cJSON_AddNumberToObject(end, "character", d->col + 1);

        cJSON_AddItemToObject(range, "start", start);
        cJSON_AddItemToObject(range, "end", end);

        cJSON_AddItemToObject(diag, "range", range);
        cJSON_AddNumberToObject(diag, "severity", 1);
        cJSON_AddStringToObject(diag, "message", d->message);

        cJSON_AddItemToArray(diag_array, diag);

        d = d->next;
    }

    cJSON_AddItemToObject(params, "diagnostics", diag_array);
    cJSON_AddItemToObject(root, "params", params);

    send_json_response(root);

    Diagnostic *cur = diagnostics.head;
    while (cur)
    {
        Diagnostic *next = cur->next;
        free(cur->message);
        free(cur);
        cur = next;
    }
}

void lsp_goto_definition(const char *uri, int line, int col, int id)
{
    ProjectFile *pf = lsp_project_get_file(uri);
    LSPIndex *idx = pf ? pf->index : NULL;

    if (!idx)
    {
        return;
    }

    LSPRange *r = lsp_find_at(idx, line, col);
    const char *target_uri = uri;
    int target_start_line = 0, target_start_col = 0;
    int target_end_line = 0, target_end_col = 0;
    int found = 0;

    if (r)
    {
        if (r->type == RANGE_DEFINITION)
        {
            target_start_line = r->start_line;
            target_start_col = r->start_col;
            target_end_line = r->end_line;
            target_end_col = r->end_col;
            found = 1;
        }
        else if (r->type == RANGE_REFERENCE && r->def_line >= 0)
        {
            LSPRange *def = lsp_find_at(idx, r->def_line, r->def_col);
            int is_local = 0;
            if (def && def->type == RANGE_DEFINITION)
            {
                is_local = 1;
            }

            if (is_local)
            {
                target_start_line = r->def_line;
                target_start_col = r->def_col;
                target_end_line = r->def_line;
                target_end_col = r->def_col; // approx
                found = 1;
            }
        }
    }

    if (!found && r && r->node)
    {
        char *name = NULL;
        if (r->node->type == NODE_EXPR_VAR)
        {
            name = r->node->var_ref.name;
        }
        else if (r->node->type == NODE_EXPR_CALL && r->node->call.callee->type == NODE_EXPR_VAR)
        {
            name = r->node->call.callee->var_ref.name;
        }

        if (name)
        {
            DefinitionResult def = lsp_project_find_definition(name);
            if (def.uri && def.range)
            {
                target_uri = def.uri;
                target_start_line = def.range->start_line;
                target_start_col = def.range->start_col;
                target_end_line = def.range->end_line;
                target_end_col = def.range->end_col;
                found = 1;
            }
        }
    }

    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "jsonrpc", "2.0");
    cJSON_AddNumberToObject(root, "id", id);

    if (found)
    {
        cJSON *result = cJSON_CreateObject();
        cJSON_AddStringToObject(result, "uri", target_uri);

        cJSON *range = cJSON_CreateObject();
        cJSON *start = cJSON_CreateObject();
        cJSON_AddNumberToObject(start, "line", target_start_line);
        cJSON_AddNumberToObject(start, "character", target_start_col);

        cJSON *end = cJSON_CreateObject();
        cJSON_AddNumberToObject(end, "line", target_end_line);
        cJSON_AddNumberToObject(end, "character", target_end_col);

        cJSON_AddItemToObject(range, "start", start);
        cJSON_AddItemToObject(range, "end", end);
        cJSON_AddItemToObject(result, "range", range);

        cJSON_AddItemToObject(root, "result", result);
    }
    else
    {
        cJSON_AddNullToObject(root, "result");
    }

    send_json_response(root);
}

void lsp_hover(const char *uri, int line, int col, int id)
{
    (void)uri;
    ProjectFile *pf = lsp_project_get_file(uri);
    LSPIndex *idx = pf ? pf->index : NULL;

    if (!idx)
    {
        return;
    }

    LSPRange *r = lsp_find_at(idx, line, col);
    char *text = NULL;

    if (r)
    {
        if (r->type == RANGE_DEFINITION)
        {
            text = r->hover_text;
        }
        else if (r->type == RANGE_REFERENCE)
        {
            LSPRange *def = lsp_find_at(idx, r->def_line, r->def_col);
            if (def && def->type == RANGE_DEFINITION)
            {
                text = def->hover_text;
            }
        }
    }

    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "jsonrpc", "2.0");
    cJSON_AddNumberToObject(root, "id", id);

    if (text)
    {
        cJSON *result = cJSON_CreateObject();
        cJSON *contents = cJSON_CreateObject();
        cJSON_AddStringToObject(contents, "kind", "markdown");

        // Need to wrap in ```c code block
        char *code_block = malloc(strlen(text) + 16);
        sprintf(code_block, "```c\n%s\n```", text);
        cJSON_AddStringToObject(contents, "value", code_block);
        free(code_block);

        cJSON_AddItemToObject(result, "contents", contents);
        cJSON_AddItemToObject(root, "result", result);
    }
    else
    {
        cJSON_AddNullToObject(root, "result");
    }

    send_json_response(root);
}

// Helper to find local in function
static ASTNode *find_local_in_func(ASTNode *func, const char *name)
{
    if (!func || !func->func.body)
    {
        return NULL;
    }

    ASTNode *queue[1024];
    int q_head = 0, q_tail = 0;
    queue[q_tail++] = func->func.body;

    // Also check params!
    if (func->func.param_names && func->func.arg_types)
    {
        for (int i = 0; i < func->func.arg_count; i++)
        {
            if (strcmp(func->func.param_names[i], name) == 0)
            {
                return NULL; // TODO: Args handled elsewhere
            }
        }
    }

    while (q_head < q_tail && q_tail < 1024)
    {
        ASTNode *curr = queue[q_head++];
        if (curr->type == NODE_VAR_DECL)
        {
            if (strcmp(curr->var_decl.name, name) == 0)
            {
                return curr;
            }
        }
        else if (curr->type == NODE_BLOCK)
        {
            ASTNode *stmt = curr->block.statements;
            while (stmt)
            {
                if (q_tail < 1024)
                {
                    queue[q_tail++] = stmt;
                }
                stmt = stmt->next;
            }
        }
        else if (curr->type == NODE_IF)
        {
            if (curr->if_stmt.then_body && q_tail < 1024)
            {
                queue[q_tail++] = curr->if_stmt.then_body;
            }
            if (curr->if_stmt.else_body && q_tail < 1024)
            {
                queue[q_tail++] = curr->if_stmt.else_body;
            }
        }
        else if (curr->type == NODE_WHILE)
        {
            if (curr->while_stmt.body && q_tail < 1024)
            {
                queue[q_tail++] = curr->while_stmt.body;
            }
        }
        else if (curr->type == NODE_FOR)
        {
            if (curr->for_stmt.init && q_tail < 1024)
            {
                queue[q_tail++] = curr->for_stmt.init;
            }
            if (curr->for_stmt.body && q_tail < 1024)
            {
                queue[q_tail++] = curr->for_stmt.body;
            }
        }
    }
    return NULL;
}

// Helper to resolve type of local var or arg
static Type *resolve_local_type(ASTNode *func, const char *name)
{
    if (!func)
    {
        return NULL;
    }
    // Check args
    if (func->func.param_names && func->func.arg_types)
    {
        for (int i = 0; i < func->func.arg_count; i++)
        {
            if (strcmp(func->func.param_names[i], name) == 0)
            {
                return func->func.arg_types[i];
            }
        }
    }
    // Check body vars
    ASTNode *decl = find_local_in_func(func, name);
    if (decl && decl->type == NODE_VAR_DECL)
    {
        return decl->var_decl.type_info;
    }
    return NULL;
}

void lsp_completion(const char *uri, int line, int col, int id)
{
    ProjectFile *pf = lsp_project_get_file(uri);
    if (!g_project || !g_project->ctx || !pf)
    {
        return;
    }

    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "jsonrpc", "2.0");
    cJSON_AddNumberToObject(root, "id", id);
    cJSON *items = cJSON_CreateArray();

    ASTNode *target_func = NULL;
    int best_line = -1;
    if (pf->index)
    {
        LSPRange *r = pf->index->head;
        while (r)
        {
            if (r->type == RANGE_DEFINITION && r->node && r->node->type == NODE_FUNCTION)
            {
                if (r->start_line <= line && r->start_line > best_line)
                {
                    best_line = r->start_line;
                    target_func = r->node;
                }
            }
            r = r->next;
        }
    }

    int dot_completed = 0;

    if (pf->source)
    {
        int cur_line = 0;
        char *ptr = pf->source;
        while (*ptr && cur_line < line)
        {
            if (*ptr == '\n')
            {
                cur_line++;
            }
            ptr++;
        }

        if (col > 0 && ptr[col - 1] == '.')
        {
            int i = col - 2;
            while (i >= 0 && (ptr[i] == ' ' || ptr[i] == '\t'))
            {
                i--;
            }
            if (i >= 0)
            {
                int end_ident = i;
                while (i >= 0 && (isalnum(ptr[i]) || ptr[i] == '_'))
                {
                    i--;
                }
                int start_ident = i + 1;

                if (start_ident <= end_ident)
                {
                    int len = end_ident - start_ident + 1;
                    char var_name[256];
                    strncpy(var_name, ptr + start_ident, len);
                    var_name[len] = 0;

                    Type *local_type = resolve_local_type(target_func, var_name);
                    char *type_name = NULL;

                    if (local_type)
                    {
                        type_name = type_to_string(local_type);
                    }
                    else
                    {
                        ASTNode *decl = find_local_in_func(target_func, var_name);
                        if (decl && decl->type == NODE_VAR_DECL && decl->var_decl.type_str)
                        {
                            type_name = strdup(decl->var_decl.type_str);
                        }
                        else
                        {
                            ZenSymbol *sym = find_symbol_in_all(g_project->ctx, var_name);
                            if (sym)
                            {
                                if (sym->type_info)
                                {
                                    type_name = type_to_string(sym->type_info);
                                }
                                else if (sym->type_name)
                                {
                                    type_name = strdup(sym->type_name);
                                }
                            }
                        }
                    }

                    if (type_name)
                    {
                        char clean_name[256];
                        char *src = type_name;
                        if (strncmp(src, "struct ", 7) == 0)
                        {
                            src += 7;
                        }

                        char *dst = clean_name;
                        while (*src && *src != '*')
                        {
                            *dst++ = *src++;
                        }
                        *dst = 0;

                        StructDef *sd = g_project->ctx->struct_defs;
                        while (sd)
                        {
                            if (strcmp(sd->name, clean_name) == 0)
                            {
                                if (sd->node && sd->node->strct.fields)
                                {
                                    ASTNode *field = sd->node->strct.fields;
                                    while (field)
                                    {
                                        cJSON *item = cJSON_CreateObject();
                                        cJSON_AddStringToObject(item, "label", field->field.name);
                                        cJSON_AddNumberToObject(item, "kind", 5); // Field
                                        char detail[256];
                                        sprintf(detail, "field %s", field->field.type);
                                        cJSON_AddStringToObject(item, "detail", detail);
                                        cJSON_AddItemToArray(items, item);
                                        field = field->next;
                                    }
                                }
                                dot_completed = 1;
                                break;
                            }
                            sd = sd->next;
                        }
                        free(type_name);
                    }
                }
            }
        }
    }

    if (!dot_completed)
    {
        const char *keywords[] = {
            "fn",     "struct",   "enum", "alias",  "return", "if",     "else",   "for",    "while",
            "break",  "continue", "true", "false",  "int",    "char",   "bool",   "string", "void",
            "import", "module",   "test", "assert", "defer",  "sizeof", "opaque", "unsafe", "asm",
            "trait",  "impl",     "u8",   "u16",    "u32",    "u64",    "i8",     "i16",    "i32",
            "i64",    "f32",      "f64",  "usize",  "isize",  "const",  "let",    "mut",    NULL};

        for (int i = 0; keywords[i]; i++)
        {
            cJSON *item = cJSON_CreateObject();
            cJSON_AddStringToObject(item, "label", keywords[i]);
            cJSON_AddNumberToObject(item, "kind", 14);
            cJSON_AddItemToArray(items, item);
        }

        StructRef *g = g_project->ctx->parsed_globals_list;
        while (g)
        {
            if (g->node)
            {
                cJSON *item = cJSON_CreateObject();
                char *name = g->node->var_decl.name;
                cJSON_AddStringToObject(item, "label", name);
                cJSON_AddNumberToObject(item, "kind", 21);
                cJSON_AddItemToArray(items, item);
            }
            g = g->next;
        }

        StructDef *s = g_project->ctx->struct_defs;
        while (s)
        {
            cJSON *item = cJSON_CreateObject();
            cJSON_AddStringToObject(item, "label", s->name);
            cJSON_AddNumberToObject(item, "kind", 22);
            cJSON_AddItemToArray(items, item);
            s = s->next;
        }

        FuncSig *f = g_project->ctx->func_registry;
        while (f)
        {
            cJSON *item = cJSON_CreateObject();
            cJSON_AddStringToObject(item, "label", f->name);
            cJSON_AddNumberToObject(item, "kind", 3);
            cJSON_AddItemToArray(items, item);
            f = f->next;
        }

        if (target_func)
        {
            if (target_func->func.param_names)
            {
                for (int i = 0; i < target_func->func.arg_count; i++)
                {
                    cJSON *item = cJSON_CreateObject();
                    cJSON_AddStringToObject(item, "label", target_func->func.param_names[i]);
                    cJSON_AddNumberToObject(item, "kind", 6);
                    cJSON_AddStringToObject(item, "detail", "argument");
                    cJSON_AddItemToArray(items, item);
                }
            }
            ASTNode *queue[1024];
            int q_head = 0;
            int q_tail = 0;
            if (target_func->func.body)
            {
                queue[q_tail++] = target_func->func.body;
            }

            while (q_head < q_tail && q_tail < 1024)
            {
                ASTNode *curr = queue[q_head++];
                if (curr->type == NODE_BLOCK)
                {
                    ASTNode *stmt = curr->block.statements;
                    while (stmt)
                    {
                        if (q_tail < 1024)
                        {
                            queue[q_tail++] = stmt;
                        }
                        stmt = stmt->next;
                    }
                }
                else if (curr->type == NODE_VAR_DECL)
                {
                    if (curr->line < line)
                    {
                        cJSON *item = cJSON_CreateObject();
                        cJSON_AddStringToObject(item, "label", curr->var_decl.name);
                        cJSON_AddNumberToObject(item, "kind", 6);
                        cJSON_AddItemToArray(items, item);
                    }
                }
                else if (curr->type == NODE_IF)
                {
                    if (curr->if_stmt.then_body && q_tail < 1024)
                    {
                        queue[q_tail++] = curr->if_stmt.then_body;
                    }
                    if (curr->if_stmt.else_body && q_tail < 1024)
                    {
                        queue[q_tail++] = curr->if_stmt.else_body;
                    }
                }
                else if (curr->type == NODE_WHILE)
                {
                    if (curr->while_stmt.body && q_tail < 1024)
                    {
                        queue[q_tail++] = curr->while_stmt.body;
                    }
                }
                else if (curr->type == NODE_FOR)
                {
                    if (curr->for_stmt.init && q_tail < 1024)
                    {
                        queue[q_tail++] = curr->for_stmt.init;
                    }
                    if (curr->for_stmt.body && q_tail < 1024)
                    {
                        queue[q_tail++] = curr->for_stmt.body;
                    }
                }
            }
        }
    }

    cJSON_AddItemToObject(root, "result", items);
    send_json_response(root);
}

static cJSON *ast_to_symbol(ASTNode *node)
{
    if (!node)
    {
        return NULL;
    }
    char *name = NULL;
    int kind = 0;

    if (node->type == NODE_FUNCTION)
    {
        name = node->func.name;
        kind = 12;
    }
    else if (node->type == NODE_STRUCT)
    {
        name = node->strct.name;
        kind = 23;
    }
    else if (node->type == NODE_VAR_DECL)
    {
        name = node->var_decl.name;
        kind = 13;
    }
    else if (node->type == NODE_CONST)
    {
        name = node->var_decl.name;
        kind = 14;
    }
    else if (node->type == NODE_ENUM)
    {
        name = node->enm.name;
        kind = 10;
    }
    else if (node->type == NODE_FIELD)
    {
        name = node->field.name;
        kind = 8;
    }

    if (!name)
    {
        return NULL;
    }

    cJSON *item = cJSON_CreateObject();
    cJSON_AddStringToObject(item, "name", name);
    cJSON_AddNumberToObject(item, "kind", kind);

    cJSON *range = cJSON_CreateObject();
    cJSON *start = cJSON_CreateObject();
    cJSON_AddNumberToObject(start, "line", node->token.line > 0 ? node->token.line - 1 : 0);
    cJSON_AddNumberToObject(start, "character", node->token.col > 0 ? node->token.col - 1 : 0);

    cJSON *end = cJSON_CreateObject();
    cJSON_AddNumberToObject(end, "line", node->token.line > 0 ? node->token.line - 1 : 0);
    cJSON_AddNumberToObject(end, "character",
                            (node->token.col > 0 ? node->token.col - 1 : 0) + node->token.len);

    cJSON_AddItemToObject(range, "start", start);
    cJSON_AddItemToObject(range, "end", end);

    cJSON_AddItemToObject(item, "range", range);
    cJSON *selRange = cJSON_Duplicate(range, 1);
    cJSON_AddItemToObject(item, "selectionRange", selRange);

    if (node->type == NODE_STRUCT && node->strct.fields)
    {
        cJSON *children = cJSON_CreateArray();
        ASTNode *f = node->strct.fields;
        while (f)
        {
            cJSON *child = ast_to_symbol(f);
            if (child)
            {
                cJSON_AddItemToArray(children, child);
            }
            f = f->next;
        }
        cJSON_AddItemToObject(item, "children", children);
    }

    return item;
}

void lsp_document_symbol(const char *uri, int id)
{
    ProjectFile *pf = lsp_project_get_file(uri);
    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "jsonrpc", "2.0");
    cJSON_AddNumberToObject(root, "id", id);

    if (!pf || !pf->ast)
    {
        cJSON_AddNullToObject(root, "result");
        send_json_response(root);
        return;
    }

    cJSON *items = cJSON_CreateArray();
    ASTNode *node = pf->ast;

    // Unwrap ROOT node if present
    if (node && node->type == NODE_ROOT)
    {
        node = node->root.children;
    }

    while (node)
    {
        cJSON *s = ast_to_symbol(node);
        if (s)
        {
            cJSON_AddItemToArray(items, s);
        }
        node = node->next; // Top level siblings
    }

    cJSON_AddItemToObject(root, "result", items);
    send_json_response(root);
}

void lsp_references(const char *uri, int line, int col, int id)
{
    ProjectFile *pf = lsp_project_get_file(uri);
    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "jsonrpc", "2.0");
    cJSON_AddNumberToObject(root, "id", id);
    cJSON *items = cJSON_CreateArray();

    if (pf && pf->index)
    {
        LSPRange *r = lsp_find_at(pf->index, line, col);
        if (r && r->node)
        {
            char *name = NULL;
            if (r->node->type == NODE_FUNCTION)
            {
                name = r->node->func.name;
            }
            else if (r->node->type == NODE_VAR_DECL)
            {
                name = r->node->var_decl.name;
            }
            else if (r->node->type == NODE_CONST)
            {
                name = r->node->var_decl.name;
            }
            else if (r->node->type == NODE_STRUCT)
            {
                name = r->node->strct.name;
            }
            else if (r->node->type == NODE_EXPR_VAR)
            {
                name = r->node->var_ref.name;
            }
            else if (r->node->type == NODE_EXPR_CALL && r->node->call.callee->type == NODE_EXPR_VAR)
            {
                name = r->node->call.callee->var_ref.name;
            }

            if (name)
            {
                ReferenceResult *refs = lsp_project_find_references(name);
                ReferenceResult *curr = refs;
                while (curr)
                {
                    cJSON *item = cJSON_CreateObject();
                    cJSON_AddStringToObject(item, "uri", curr->uri);
                    cJSON *range = cJSON_CreateObject();
                    cJSON *start = cJSON_CreateObject();
                    cJSON_AddNumberToObject(start, "line", curr->range->start_line);
                    cJSON_AddNumberToObject(start, "character", curr->range->start_col);

                    cJSON *end = cJSON_CreateObject();
                    cJSON_AddNumberToObject(end, "line", curr->range->end_line);
                    cJSON_AddNumberToObject(end, "character", curr->range->end_col);

                    cJSON_AddItemToObject(range, "start", start);
                    cJSON_AddItemToObject(range, "end", end);
                    cJSON_AddItemToObject(item, "range", range);
                    cJSON_AddItemToArray(items, item);

                    ReferenceResult *next = curr->next;
                    free(curr);
                    curr = next;
                }
            }
        }
    }

    cJSON_AddItemToObject(root, "result", items);
    send_json_response(root);
}

void lsp_signature_help(const char *uri, int line, int col, int id)
{
    ProjectFile *pf = lsp_project_get_file(uri);
    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "jsonrpc", "2.0");
    cJSON_AddNumberToObject(root, "id", id);

    if (!g_project || !g_project->ctx || !pf || !pf->source)
    {
        cJSON_AddNullToObject(root, "result");
        send_json_response(root);
        return;
    }

    // ... [Scan backwards logic same as before] ...
    char *ptr = pf->source;
    int cur_line = 0;
    while (*ptr && cur_line < line)
    {
        if (*ptr == '\n')
        {
            cur_line++;
        }
        ptr++;
    }
    if (ptr && col > 0)
    {
        ptr += col;
    }

    if (ptr > pf->source + strlen(pf->source))
    {
        cJSON_AddNullToObject(root, "result");
        send_json_response(root);
        return;
    }

    int found = 0;
    char *p = ptr - 1;
    while (p >= pf->source)
    {
        if (*p == ')')
        {
            break;
        }
        if (*p == '(')
        {
            // Found open paren
            char *ident_end = p - 1;
            while (ident_end >= pf->source && isspace(*ident_end))
            {
                ident_end--;
            }
            if (ident_end < pf->source)
            {
                break;
            }
            char *ident_start = ident_end;
            while (ident_start >= pf->source && (isalnum(*ident_start) || *ident_start == '_'))
            {
                ident_start--;
            }
            ident_start++;

            int len = ident_end - ident_start + 1;
            if (len > 0 && len < 255)
            {
                char func_name[256];
                strncpy(func_name, ident_start, len);
                func_name[len] = 0;
                // Lookup
                FuncSig *fn = g_project->ctx->func_registry;
                while (fn)
                {
                    if (strcmp(fn->name, func_name) == 0)
                    {
                        // Found it
                        cJSON *result = cJSON_CreateObject();
                        cJSON *sigs = cJSON_CreateArray();
                        cJSON *sig = cJSON_CreateObject();

                        char label[2048];
                        char params[1024] = "";
                        int first = 1;
                        for (int i = 0; i < fn->total_args; i++)
                        {
                            if (!first)
                            {
                                strcat(params, ", ");
                            }
                            char *tstr = type_to_string(fn->arg_types[i]);
                            if (tstr)
                            {
                                strcat(params, tstr);
                                free(tstr);
                            }
                            else
                            {
                                strcat(params, "unknown");
                            }
                            first = 0;
                        }
                        char *ret_str = type_to_string(fn->ret_type);
                        sprintf(label, "fn %s(%s) -> %s", fn->name, params,
                                ret_str ? ret_str : "void");
                        if (ret_str)
                        {
                            free(ret_str);
                        }

                        cJSON_AddStringToObject(sig, "label", label);
                        cJSON_AddItemToObject(sig, "parameters", cJSON_CreateArray());
                        cJSON_AddItemToArray(sigs, sig);

                        cJSON_AddItemToObject(result, "signatures", sigs);
                        cJSON_AddItemToObject(root, "result", result);
                        found = 1;
                        break;
                    }
                    fn = fn->next;
                }
            }
            break;
        }
        p--;
    }

    if (!found)
    {
        cJSON_AddNullToObject(root, "result");
    }

    send_json_response(root);
}

static char *get_symbol_at(ProjectFile *pf, int line, int col)
{
    if (!pf || !pf->index)
    {
        return NULL;
    }
    LSPRange *r = pf->index->head;
    while (r)
    {
        int over_start = (line > r->start_line) || (line == r->start_line && col >= r->start_col);
        int under_end = (line < r->end_line) || (line == r->end_line && col <= r->end_col);

        if (over_start && under_end && r->node)
        {
            if (r->node->type == NODE_FUNCTION)
            {
                return strdup(r->node->func.name);
            }
            if (r->node->type == NODE_VAR_DECL)
            {
                return strdup(r->node->var_decl.name);
            }
            if (r->node->type == NODE_CONST)
            {
                return strdup(r->node->var_decl.name);
            }
            if (r->node->type == NODE_STRUCT)
            {
                return strdup(r->node->strct.name);
            }
            if (r->node->type == NODE_EXPR_VAR)
            {
                return strdup(r->node->var_ref.name);
            }
            if (r->node->type == NODE_EXPR_CALL)
            {
                if (r->node->call.callee && r->node->call.callee->type == NODE_EXPR_VAR)
                {
                    return strdup(r->node->call.callee->var_ref.name);
                }
            }
        }
        r = r->next;
    }
    return NULL;
}

void lsp_rename(const char *uri, int line, int col, const char *new_name, int id)
{
    ProjectFile *pf = lsp_project_get_file(uri);
    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "jsonrpc", "2.0");
    cJSON_AddNumberToObject(root, "id", id);

    char *name = get_symbol_at(pf, line, col);
    if (!name)
    {
        cJSON_AddNullToObject(root, "result");
        send_json_response(root);
        return;
    }

    ReferenceResult *refs = lsp_project_find_references(name);
    free(name);

    if (!refs)
    {
        cJSON_AddNullToObject(root, "result");
        send_json_response(root);
        return;
    }

    cJSON *result = cJSON_CreateObject();
    cJSON *changes = cJSON_CreateObject();

    ReferenceResult *cur = refs;
    while (cur)
    {
        cJSON *edits = cJSON_GetObjectItem(changes, cur->uri);
        if (!edits)
        {
            edits = cJSON_CreateArray();
            cJSON_AddItemToObject(changes, cur->uri, edits);
        }

        cJSON *edit = cJSON_CreateObject();
        cJSON *range = cJSON_CreateObject();
        cJSON *start = cJSON_CreateObject();
        cJSON *end = cJSON_CreateObject();

        cJSON_AddNumberToObject(start, "line", cur->range->start_line);
        cJSON_AddNumberToObject(start, "character", cur->range->start_col);
        cJSON_AddNumberToObject(end, "line", cur->range->end_line);
        cJSON_AddNumberToObject(end, "character", cur->range->end_col);

        cJSON_AddItemToObject(range, "start", start);
        cJSON_AddItemToObject(range, "end", end);
        cJSON_AddItemToObject(edit, "range", range);
        cJSON_AddStringToObject(edit, "newText", new_name);

        cJSON_AddItemToArray(edits, edit);

        ReferenceResult *next = cur->next;
        free(cur);
        cur = next;
    }

    cJSON_AddItemToObject(result, "changes", changes);
    cJSON_AddItemToObject(root, "result", result);
    send_json_response(root);
}
