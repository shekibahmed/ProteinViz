---
title: Streamlit st.navigation breaks from a pages/ directory and entry-relative page_link paths
date: 2026-10-03
category: ui-bugs
module: web-app
problem_type: ui_bug
component: frontend
symptoms:
  - "Sidebar lists raw file names (streamlit app, about, disease…) instead of the st.navigation sections"
  - "set_page_config and sidebar captions from the entry script never appear on deep links"
  - "StreamlitPageNotFoundError: Could not find page: views/disease.py when launched from a different entry script"
root_cause: wrong_api
resolution_type: code_fix
severity: medium
framework_version: streamlit 1.65.0
tags: [streamlit, st-navigation, st-page-link, multipage, entry-point, pages-directory]
---

# Streamlit st.navigation breaks from a pages/ directory and entry-relative page_link paths

## Problem
The app uses `st.navigation` with page files next to the entry script. Two separate Streamlit
path conventions broke navigation in ways the in-process test harness did not catch (a related
default-page gotcha is covered in Solution step 3):
1. a folder named `pages/` beside the entry script switched on Streamlit's legacy automatic
   multipage mode;
2. `st.page_link("views/x.py")` resolved relative to *whichever script launched the app*, so it
   crashed once a second entry point (the repo-root `streamlit_app.py` used by Community Cloud)
   was added.

## Symptoms
- In a real browser, the sidebar showed file names ("streamlit app", "about", "disease"…), deep
  links such as `/structure` ran the page file directly, and the entry script's
  `st.set_page_config(layout="wide")` and sidebar captions were missing.
- Launched via `streamlit run streamlit_app.py` from the repo root, Home crashed with
  `StreamlitPageNotFoundError: Could not find page: views/disease.py. You must provide a Page
  object or file path relative to the entrypoint file.`
- `streamlit.testing.v1.AppTest` tests passed in both cases, because they started from the
  package entry script and used `switch_page`.

## What Didn't Work
- **Relying on AppTest alone.** It ran the pages with the expected entry script and never went
  through the browser URL routing, so neither bug showed up. Both were found by loading the app in
  headless Chromium (Playwright) and looking at the screenshots.
- **Renaming the folder by itself was not enough for the second bug.** After the page folder was
  renamed to `src/proteinviz/app/views/`, string paths in `page_link` still depended on the launching script's location.

## Solution
1. **Never name the page folder `pages/` when using `st.navigation`.** Pages live in
   `src/proteinviz/app/views/`.
2. **Define pages once and link to `st.Page` objects, not file-path strings.**
   `src/proteinviz/app/navigation.py` holds the single table of pages:
   ```python
   VIEWS = Path(__file__).parent / "views"          # absolute, independent of the entry script

   def page(name: str) -> st.Page:
       title, icon, url_path, _ = PAGES[name]
       if url_path is None:
           return st.Page(VIEWS / f"{name}.py", title=title, icon=icon, default=True)
       return st.Page(VIEWS / f"{name}.py", title=title, icon=icon, url_path=url_path)
   ```
   The entry script calls `st.navigation(sections())` (`src/proteinviz/app/streamlit_app.py:12`),
   and views link with the same constructor:
   ```python
   # before: breaks when launched from another entry script
   st.page_link("views/disease.py", label="Disease explorer", icon="🩺")
   # after
   st.page_link(page("disease"), label="Disease explorer", icon="🩺")
   ```
3. Don't give the default page a `url_path`: it is always served at `/`, and `/home` returned
   "Page not found".

Steps 1 and 3 were caught in a browser check while PR #2 was being developed, so the `pages/`
folder and the `/home` URL never reached git history. Step 2 (`navigation.py` and page-object
links) landed in PR #5.

## Why This Works
Streamlit's legacy multipage mode auto-discovers a directory literally named `pages` next to the
main script. Its routing then competes with `st.navigation`, and URL deep links bypass the
entry script entirely. Any other folder name avoids that discovery.

`st.page_link` accepts a `st.Page` or a path *relative to the main script*. A `st.Page` built
from an absolute path matches the page registered with `st.navigation`, whichever file was
passed to `streamlit run`. A single page table also keeps titles, icons and `url_path`s from
drifting between the navigation and the links, which is what makes permalinks stable.

## Prevention
- An AppTest that renders Home from **every** entry script (`tests/test_app.py`,
  `test_home_links_resolve_from_any_entry_point`, parametrized over the package entry and the
  repo-root `CLOUD_ENTRY`). It failed on the old string paths and passes now.
- After UI changes, load real URLs in a browser, including deep links like
  `/structure?a=TP53&b=MDM2`. In-process AppTest does not exercise URL routing.
- Keep page definitions in one module, and never hand-write view file paths in views.

## Related Issues
- `docs/solutions/integration-issues/streamlit-community-cloud-skips-extras-from-uv-lock.md`:
  the same deployment change (a second, repo-root entry point) exposed the `page_link` bug
- PR #2, PR #5
