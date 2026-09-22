# Capturing the Phoenix UI

The integration is verified programmatically (`tests/`, and the client read-back
shown in the README), but a screenshot makes it tangible for a reader. These are
the exact steps, and where the image belongs.

1. `phoenix serve`
2. `culprit demo --project culprit-demo`
3. Open <http://localhost:6006/projects> and click `culprit-demo`.
4. In the span table, click any row whose **status** is red (a failed run).
5. In the trace tree on the right, click through the spans. The one `culprit`
   accused shows **Annotations +1**; expanding it shows `culprit  1.00` with the
   explanation.
6. Capture it (macOS: `cmd+shift+4`, then drag) and save as
   `docs/phoenix-annotation.png`.
7. The README references it at that path.

A good frame includes three things at once: the span tree, the selected span's
input/output, and the `culprit` annotation on it. That is the whole claim of the
integration in one image — the verdict lives on the span it accuses, inside the
tool the engineer already uses.
