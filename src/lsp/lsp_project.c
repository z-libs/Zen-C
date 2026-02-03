#include "lsp_project.h"
//#include <dirent.h>
#include "compat/compat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

LSPProject *g_project = NULL;

static void scan_dir(const char *dir_path);

void lsp_project_init(const char *root_path)
{
    if (g_project)
    {
        return;
    }

    fprintf(stderr, "zls: Initializing project at %s\n", root_path);

    g_project = xcalloc(1, sizeof(LSPProject));
    g_project->root_path = xstrdup(root_path);

    // Create a persistent global context
    g_project->ctx = xcalloc(1, sizeof(ParserContext));
    g_project->ctx->is_fault_tolerant = 1;

    // Set a default error handler that just logs to stderr (or ignores)
    // to prevent exit(1) during initial scan.
    void lsp_default_on_error(void *data, Token t, const char *msg);
    g_project->ctx->on_error = lsp_default_on_error;

    // Scan workspace
    scan_dir(root_path);
}

// Default error handler for indexing phase
void lsp_default_on_error(void *data, Token t, const char *msg)
{
    (void)data;
    (void)t;
    (void)msg;
    // We can log it if we want, but standard zpanic_at already printed it to stderr.
    // The important thing is that we exist so zpanic_at returns.
    // Maybe we suppress duplicates or just let it pass.
    // Since zpanic_at printed "error: ...", we don't need to print again.
}

static void scan_file(const char *path)
{
    // Skip if not .zc
    const char *ext = strrchr(path, '.');
    if (!ext || strcmp(ext, ".zc") != 0)
    {
        return;
    }

    fprintf(stderr, "zls: Indexing %s\n", path);

    char *src = load_file(path);
    if (!src)
    {
        return;
    }

    char uri[2048];
    sprintf(uri, "file://%s", path);

    lsp_project_update_file(uri, src);
    free(src);
}

static void scan_dir(const char *dir_path)
{
    ZCDir *d = zc_opendir(dir_path);
    if (!d)
    {
        return;
    }

    const ZCDirEnt *dir;
    while ((dir = zc_readdir(d)) != NULL)
    {
        if (dir->name[0] == '.')
        {
            continue;
        }

        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir_path, dir->name);

        if (zc_is_dir(path))
        {
            scan_dir(path);
        }
        else
        {
            scan_file(path);
        }
    }
    zc_closedir(d);
}

ProjectFile *lsp_project_get_file(const char *uri)
{
    if (!g_project)
    {
        return NULL;
    }
    ProjectFile *curr = g_project->files;
    while (curr)
    {
        if (strcmp(curr->uri, uri) == 0)
        {
            return curr;
        }
        curr = curr->next;
    }
    return NULL;
}

static ProjectFile *add_project_file(const char *uri)
{
    ProjectFile *f = xcalloc(1, sizeof(ProjectFile));
    f->uri = xstrdup(uri);
    // Simple path extraction from URI (file://...)
    if (strncmp(uri, "file://", 7) == 0)
    {
        f->path = xstrdup(uri + 7);
    }
    else
    {
        f->path = xstrdup(uri);
    }

    f->next = g_project->files;
    g_project->files = f;
    return f;
}

void lsp_project_update_file(const char *uri, const char *src)
{
    if (!g_project)
    {
        return;
    }

    ProjectFile *pf = lsp_project_get_file(uri);
    if (!pf)
    {
        pf = add_project_file(uri);
    }

    // Clear old index
    if (pf->index)
    {
        lsp_index_free(pf->index);
        pf->index = NULL;
    }

    // Update source
    if (pf->source)
    {
        free(pf->source);
    }
    pf->source = xstrdup(src);

    // Parse
    Lexer l;
    lexer_init(&l, src);

    ASTNode *root = parse_program(g_project->ctx, &l);

    // Build Index
    pf->index = lsp_index_new();
    if (root)
    {
        lsp_build_index(pf->index, root);
    }
}

DefinitionResult lsp_project_find_definition(const char *name)
{
    // ... existing implementation ...
    DefinitionResult res = {0};
    if (!g_project)
    {
        return res;
    }

    ProjectFile *pf = g_project->files;
    while (pf)
    {
        if (pf->index)
        {
            LSPRange *r = pf->index->head;
            while (r)
            {
                if (r->type == RANGE_DEFINITION && r->node)
                {
                    // Check name match
                    char *found_name = NULL;
                    if (r->node->type == NODE_FUNCTION)
                    {
                        found_name = r->node->func.name;
                    }
                    else if (r->node->type == NODE_VAR_DECL)
                    {
                        found_name = r->node->var_decl.name;
                    }
                    else if (r->node->type == NODE_CONST)
                    {
                        found_name = r->node->var_decl.name;
                    }
                    else if (r->node->type == NODE_STRUCT)
                    {
                        found_name = r->node->strct.name;
                    }

                    if (found_name && strcmp(found_name, name) == 0)
                    {
                        res.uri = pf->uri;
                        res.range = r;
                        return res;
                    }
                }
                r = r->next;
            }
        }
        pf = pf->next;
    }

    return res;
}

// Find all references to a symbol name project-wide

ReferenceResult *lsp_project_find_references(const char *name)
{
    if (!g_project)
    {
        return NULL;
    }
    ReferenceResult *head = NULL;
    ReferenceResult *tail = NULL;

    ProjectFile *pf = g_project->files;
    while (pf)
    {
        if (pf->index)
        {
            LSPRange *r = pf->index->head;
            while (r)
            {
                // We want REFERENCES that match the name
                // Or DEFINITIONS that match the name (include decl)
                char *scan_name = NULL;

                if (r->node)
                {
                    if (r->node->type == NODE_FUNCTION)
                    {
                        scan_name = r->node->func.name;
                    }
                    else if (r->node->type == NODE_VAR_DECL)
                    {
                        scan_name = r->node->var_decl.name;
                    }
                    else if (r->node->type == NODE_CONST)
                    {
                        scan_name = r->node->var_decl.name;
                    }
                    else if (r->node->type == NODE_STRUCT)
                    {
                        scan_name = r->node->strct.name;
                    }
                    else if (r->node->type == NODE_EXPR_VAR)
                    {
                        scan_name = r->node->var_ref.name;
                    }
                    else if (r->node->type == NODE_EXPR_CALL &&
                             r->node->call.callee->type == NODE_EXPR_VAR)
                    {
                        scan_name = r->node->call.callee->var_ref.name;
                    }
                }

                if (scan_name && strcmp(scan_name, name) == 0)
                {
                    ReferenceResult *new_res = calloc(1, sizeof(ReferenceResult));
                    new_res->uri = pf->uri;
                    new_res->range = r;

                    if (!head)
                    {
                        head = new_res;
                        tail = new_res;
                    }
                    else
                    {
                        tail->next = new_res;
                        tail = new_res;
                    }
                }

                r = r->next;
            }
        }
        pf = pf->next;
    }
    return head;
}
