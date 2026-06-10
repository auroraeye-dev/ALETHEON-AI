#!/usr/bin/env python3
"""
Aletheon: fix Qdrant per-process reset + compare flow.

Idempotent — safe to run multiple times. Creates .bak files for each edit.

Run from repo root:
    python qdrant_compare_fix.py
"""
import os, sys, shutil

REPO_ROOT = os.path.expanduser("~/Desktop/aletheon_cl_code")

# ---- 1. graph.py: REMOVE the per-run reset from _index_node ----------------

GRAPH_PATH = os.path.join(REPO_ROOT, "core/graph.py")

OLD_INDEX_BLOCK = (
    "    vectors = embed_texts([c.text for c in chunks])\n"
    "    vectorstore.reset()  # per-run cleanup: each main.py invocation starts\n"
    "                         # with a fresh collection. Keeps local Qdrant fast\n"
    "                         # (warning kicks in past 20K points) and prevents\n"
    "                         # stale chunks from previous drug runs polluting\n"
    "                         # retrieval. Cache layer is separate and unaffected.\n"
    "    vectorstore.index_chunks(chunks, vectors)"
)
NEW_INDEX_BLOCK = (
    "    vectors = embed_texts([c.text for c in chunks])\n"
    "    # NOTE: collection-wide reset happens ONCE at process start (in main.py\n"
    "    # via vectorstore.clear_storage()), not per-pipeline. Per-pipeline reset\n"
    "    # was wrong for the compare flow because it wiped drug-1's chunks before\n"
    "    # drug-2's pipeline could be retrieved. Multi-drug runs co-index into\n"
    "    # the same collection and use the drug payload filter to separate.\n"
    "    vectorstore.index_chunks(chunks, vectors)"
)


# ---- 2. vectorstore.py: ADD clear_storage() helper -------------------------

VS_PATH = os.path.join(REPO_ROOT, "storage/vectorstore.py")

VS_MARKER = "def clear_storage("
VS_INSERT_AFTER = "atexit.register(_close_client)"
VS_NEW_FUNC = """


def clear_storage() -> None:
    \"\"\"Process-startup cleanup: wipe the local Qdrant storage entirely.

    Called once at CLI entry (main.py:_dispatch) before any pipeline runs.

    Why this exists:
    - The vector index is fully derived from the source cache. Wiping it
      between invocations costs nothing (re-embedding from already-cached
      evidence is fast) and gives us deterministic state every time.
    - Prevents the 20K-points local-mode warning from accumulating across
      runs.
    - Handles stale lock files left behind when a previous process was
      killed with Ctrl-C or crashed before atexit could close the client.
      Without this, the next run hits 'Storage folder ... is already
      accessed by another instance of Qdrant client.'
    - For multi-drug runs (compare), the storage is wiped ONCE at process
      start; both drug pipelines then co-index into the same collection,
      separated by the 'drug' payload filter at retrieve time. (Per-
      pipeline reset would clobber drug 1 when drug 2 indexes.)
    \"\"\"
    import shutil
    global _client
    # Close any existing client so it releases its file handles on the
    # storage directory (otherwise rmtree on macOS will keep stale handles).
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None
    if os.path.isdir(QDRANT_PATH):
        try:
            shutil.rmtree(QDRANT_PATH)
            log.info("[qdrant] storage cleared (fresh process)")
        except Exception as e:
            log.warning(f"[qdrant] storage clear failed: {e}")
    # Recreate the empty directory so subsequent _get_client() calls
    # have a valid path to open.
    os.makedirs(QDRANT_PATH, exist_ok=True)
"""


# ---- 3. main.py: CALL clear_storage() at top of _dispatch ------------------

MAIN_PATH = os.path.join(REPO_ROOT, "main.py")

MAIN_MARKER = "vectorstore.clear_storage()"
MAIN_OLD = (
    "def _dispatch(args):\n"
    "    def _parse_depth(arglist):"
)
MAIN_NEW = (
    "def _dispatch(args):\n"
    "    # PROCESS-STARTUP CLEANUP — wipe the Qdrant storage so every CLI\n"
    "    # invocation starts deterministically. Skipped only for housekeeping\n"
    "    # subcommands that don't touch the vector store (cache-clear, export).\n"
    "    # Use --keep-index to opt out (e.g. for debugging an existing index).\n"
    "    _no_clear_cmds = {\"cache-clear\", \"export\"}\n"
    "    if args and args[0] not in _no_clear_cmds and \"--keep-index\" not in args:\n"
    "        try:\n"
    "            from storage import vectorstore\n"
    "            vectorstore.clear_storage()\n"
    "        except Exception as _e:\n"
    "            # Don't block CLI on cleanup failure — log and continue.\n"
    "            from core.logging_setup import log as _log\n"
    "            _log.warning(f\"[startup] qdrant clear skipped: {_e}\")\n"
    "\n"
    "    def _parse_depth(arglist):"
)


# ---- patcher --------------------------------------------------------------

def patch_file(path: str, old: str, new: str, marker_for_already_applied: str = None,
               label: str = "") -> str:
    """Returns one of: 'applied', 'already', 'missing-target', 'file-missing'."""
    if not os.path.exists(path):
        return "file-missing"
    src = open(path).read()
    if marker_for_already_applied and marker_for_already_applied in src:
        return "already"
    if old not in src:
        return "missing-target"
    shutil.copy(path, path + ".bak")
    open(path, "w").write(src.replace(old, new, 1))
    return "applied"


print("=" * 70)
print("Aletheon: Qdrant per-process reset + compare flow fix")
print("=" * 70)
print()

results = {
    "graph.py":       patch_file(GRAPH_PATH, OLD_INDEX_BLOCK, NEW_INDEX_BLOCK,
                                 marker_for_already_applied="reset happens ONCE at process start",
                                 label="remove per-run reset from _index_node"),
    "vectorstore.py": patch_file(VS_PATH,
                                 VS_INSERT_AFTER,
                                 VS_INSERT_AFTER + VS_NEW_FUNC,
                                 marker_for_already_applied=VS_MARKER,
                                 label="add clear_storage() helper"),
    "main.py":        patch_file(MAIN_PATH, MAIN_OLD, MAIN_NEW,
                                 marker_for_already_applied=MAIN_MARKER,
                                 label="call clear_storage() at dispatch entry"),
}

for fname, status in results.items():
    flag = {"applied": "✓ APPLIED  ",
            "already": "= ALREADY  ",
            "missing-target": "✗ TARGET MISSING — file may have unexpected edits, skipped",
            "file-missing":   "✗ FILE NOT FOUND"}[status]
    print(f"  {flag} {fname}")

print()
print("Backups created at *.bak alongside each edited file.")
print()
print("Verify with:")
print("  grep -n 'reset happens ONCE' core/graph.py")
print("  grep -n 'def clear_storage' storage/vectorstore.py")
print("  grep -n 'clear_storage()' main.py")
print()
print("Rollback if needed:")
print("  cp core/graph.py.bak core/graph.py")
print("  cp storage/vectorstore.py.bak storage/vectorstore.py")
print("  cp main.py.bak main.py")