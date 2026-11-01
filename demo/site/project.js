// Preserve links shared before the guide moved to its own page.
const current = new URL(window.location.href);
if (
  ["example", "step", "standard"].some((key) => current.searchParams.has(key))
) {
  const guide = new URL("guide/", current);
  guide.search = current.search;
  guide.hash = current.hash;
  window.location.replace(guide.href);
}
