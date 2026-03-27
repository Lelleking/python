(function () {
  function ensureStyle() {
    if (document.getElementById("codex-collapsible-style")) return;
    const style = document.createElement("style");
    style.id = "codex-collapsible-style";
    style.textContent = `
      details.codex-collapsible-menu {
        border: 1px solid #d1d5db;
        border-radius: 10px;
        background: #f8fafc;
        padding: 8px;
        margin: 8px 0 12px 0;
      }
      details.codex-collapsible-menu > summary {
        cursor: pointer;
        font-weight: 700;
        color: #0f172a;
        list-style: none;
      }
      details.codex-collapsible-menu > summary::-webkit-details-marker { display: none; }
      details.codex-collapsible-menu > summary::after {
        content: "▾";
        float: right;
        color: #475569;
      }
      details.codex-collapsible-menu:not([open]) > summary::after { content: "▸"; }
      .codex-collapsible-inner { margin-top: 8px; }
    `;
    document.head.appendChild(style);
  }

  function menuLabelForForm(form) {
    const explicit = (form.getAttribute("data-collapse-label") || "").trim();
    if (explicit) return explicit;
    const h = form.querySelector("h2, h3, .pane-title, legend");
    if (h && h.textContent) return h.textContent.trim();
    return "Controls";
  }

  function shouldCollapseForm(form) {
    if (!form || form.dataset.collapseApplied === "1") return false;
    if (form.hasAttribute("data-no-auto-collapse")) return false;
    if (form.closest(".grouping-modal, .folder-policy-modal, .folder-crossed-modal, .auto-group-modal, .plate-overview-modal")) {
      return false;
    }
    const interactive = form.querySelectorAll("input, select, button, textarea, details").length;
    return interactive >= 10;
  }

  function collapseForm(form) {
    const children = Array.from(form.children || []);
    const movable = children.filter((el) => !(el.tagName === "INPUT" && String(el.type || "").toLowerCase() === "hidden"));
    if (movable.length < 3) return;

    const details = document.createElement("details");
    details.className = "codex-collapsible-menu";
    details.open = false;
    const summary = document.createElement("summary");
    summary.textContent = menuLabelForForm(form);
    details.appendChild(summary);
    const inner = document.createElement("div");
    inner.className = "codex-collapsible-inner";
    details.appendChild(inner);

    movable.forEach((el) => inner.appendChild(el));
    form.appendChild(details);
    form.dataset.collapseApplied = "1";
  }

  function run() {
    ensureStyle();
    const forms = Array.from(document.querySelectorAll("form"));
    forms.forEach((form) => {
      if (shouldCollapseForm(form)) collapseForm(form);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", run);
  } else {
    run();
  }
})();
