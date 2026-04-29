import { CatchBoundary } from "./CatchBoundary.js";
import { useRouter } from "./useRouter.js";
import { isNotFound } from "@tanstack/router-core";
import "react";
import { jsx } from "react/jsx-runtime";
import { useStore } from "@tanstack/react-store";
import { isServer } from "@tanstack/router-core/isServer";
//#region src/not-found.tsx
function CatchNotFound(props) {
	const router = useRouter();
	if (isServer ?? router.isServer) {
		const resetKey = `not-found-${router.stores.location.state.pathname}-${router.stores.status.state}`;
		return /* @__PURE__ */ jsx(CatchBoundary, {
			getResetKey: () => resetKey,
			onCatch: (error, errorInfo) => {
				if (isNotFound(error)) props.onCatch?.(error, errorInfo);
				else throw error;
			},
			errorComponent: ({ error }) => {
				if (isNotFound(error)) return props.fallback?.(error);
				else throw error;
			},
			children: props.children
		});
	}
	const resetKey = `not-found-${useStore(router.stores.location, (location) => location.pathname)}-${useStore(router.stores.status, (status) => status)}`;
	return /* @__PURE__ */ jsx(CatchBoundary, {
		getResetKey: () => resetKey,
		onCatch: (error, errorInfo) => {
			if (isNotFound(error)) props.onCatch?.(error, errorInfo);
			else throw error;
		},
		errorComponent: ({ error }) => {
			if (isNotFound(error)) return props.fallback?.(error);
			else throw error;
		},
		children: props.children
	});
}
function DefaultGlobalNotFound() {
	return /* @__PURE__ */ jsx("p", { children: "Not Found" });
}
//#endregion
export { CatchNotFound, DefaultGlobalNotFound };

//# sourceMappingURL=not-found.js.map