const require_runtime = require("./_virtual/_rolldown/runtime.cjs");
const require_ClientOnly = require("./ClientOnly.cjs");
const require_useRouter = require("./useRouter.cjs");
let react = require("react");
react = require_runtime.__toESM(react);
let react_jsx_runtime = require("react/jsx-runtime");
let _tanstack_router_core_isServer = require("@tanstack/router-core/isServer");
//#region src/Asset.tsx
function Asset({ tag, attrs, children, nonce }) {
	switch (tag) {
		case "title": return /* @__PURE__ */ (0, react_jsx_runtime.jsx)("title", {
			...attrs,
			suppressHydrationWarning: true,
			children
		});
		case "meta": return /* @__PURE__ */ (0, react_jsx_runtime.jsx)("meta", {
			...attrs,
			suppressHydrationWarning: true
		});
		case "link": return /* @__PURE__ */ (0, react_jsx_runtime.jsx)("link", {
			...attrs,
			nonce,
			suppressHydrationWarning: true
		});
		case "style": return /* @__PURE__ */ (0, react_jsx_runtime.jsx)("style", {
			...attrs,
			dangerouslySetInnerHTML: { __html: children },
			nonce
		});
		case "script": return /* @__PURE__ */ (0, react_jsx_runtime.jsx)(Script, {
			attrs,
			children
		});
		default: return null;
	}
}
function Script({ attrs, children }) {
	const router = require_useRouter.useRouter();
	const hydrated = require_ClientOnly.useHydrated();
	const dataScript = typeof attrs?.type === "string" && attrs.type !== "" && attrs.type !== "text/javascript" && attrs.type !== "module";
	if (process.env.NODE_ENV !== "production" && attrs?.src && typeof children === "string" && children.trim().length) console.warn("[TanStack Router] <Script> received both `src` and `children`. The `children` content will be ignored. Remove `children` or remove `src`.");
	react.useEffect(() => {
		if (dataScript) return;
		if (attrs?.src) {
			const normSrc = (() => {
				try {
					const base = document.baseURI || window.location.href;
					return new URL(attrs.src, base).href;
				} catch {
					return attrs.src;
				}
			})();
			if (Array.from(document.querySelectorAll("script[src]")).find((el) => el.src === normSrc)) return;
			const script = document.createElement("script");
			for (const [key, value] of Object.entries(attrs)) if (key !== "suppressHydrationWarning" && value !== void 0 && value !== false) script.setAttribute(key, typeof value === "boolean" ? "" : String(value));
			document.head.appendChild(script);
			return () => {
				if (script.parentNode) script.parentNode.removeChild(script);
			};
		}
		if (typeof children === "string") {
			const typeAttr = typeof attrs?.type === "string" ? attrs.type : "text/javascript";
			const nonceAttr = typeof attrs?.nonce === "string" ? attrs.nonce : void 0;
			if (Array.from(document.querySelectorAll("script:not([src])")).find((el) => {
				if (!(el instanceof HTMLScriptElement)) return false;
				const sType = el.getAttribute("type") ?? "text/javascript";
				const sNonce = el.getAttribute("nonce") ?? void 0;
				return el.textContent === children && sType === typeAttr && sNonce === nonceAttr;
			})) return;
			const script = document.createElement("script");
			script.textContent = children;
			if (attrs) {
				for (const [key, value] of Object.entries(attrs)) if (key !== "suppressHydrationWarning" && value !== void 0 && value !== false) script.setAttribute(key, typeof value === "boolean" ? "" : String(value));
			}
			document.head.appendChild(script);
			return () => {
				if (script.parentNode) script.parentNode.removeChild(script);
			};
		}
	}, [
		attrs,
		children,
		dataScript
	]);
	if (_tanstack_router_core_isServer.isServer ?? router.isServer) {
		if (attrs?.src) return /* @__PURE__ */ (0, react_jsx_runtime.jsx)("script", {
			...attrs,
			suppressHydrationWarning: true
		});
		if (typeof children === "string") return /* @__PURE__ */ (0, react_jsx_runtime.jsx)("script", {
			...attrs,
			dangerouslySetInnerHTML: { __html: children },
			suppressHydrationWarning: true
		});
		return null;
	}
	if (dataScript && typeof children === "string") return /* @__PURE__ */ (0, react_jsx_runtime.jsx)("script", {
		...attrs,
		suppressHydrationWarning: true,
		dangerouslySetInnerHTML: { __html: children }
	});
	if (!hydrated) {
		if (attrs?.src) return /* @__PURE__ */ (0, react_jsx_runtime.jsx)("script", {
			...attrs,
			suppressHydrationWarning: true
		});
		if (typeof children === "string") return /* @__PURE__ */ (0, react_jsx_runtime.jsx)("script", {
			...attrs,
			dangerouslySetInnerHTML: { __html: children },
			suppressHydrationWarning: true
		});
	}
	return null;
}
//#endregion
exports.Asset = Asset;

//# sourceMappingURL=Asset.cjs.map