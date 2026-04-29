import { reactUse } from "./utils.js";
import { TSR_DEFERRED_PROMISE, defer } from "@tanstack/router-core";
import * as React$1 from "react";
import { jsx } from "react/jsx-runtime";
//#region src/awaited.tsx
/** Suspend until a deferred promise resolves or rejects and return its data. */
function useAwaited({ promise: _promise }) {
	if (reactUse) return reactUse(_promise);
	const promise = defer(_promise);
	if (promise[TSR_DEFERRED_PROMISE].status === "pending") throw promise;
	if (promise[TSR_DEFERRED_PROMISE].status === "error") throw promise[TSR_DEFERRED_PROMISE].error;
	return promise[TSR_DEFERRED_PROMISE].data;
}
/**
* Component that suspends on a deferred promise and renders its child with
* the resolved value. Optionally provides a Suspense fallback.
*/
function Await(props) {
	const inner = /* @__PURE__ */ jsx(AwaitInner, { ...props });
	if (props.fallback) return /* @__PURE__ */ jsx(React$1.Suspense, {
		fallback: props.fallback,
		children: inner
	});
	return inner;
}
function AwaitInner(props) {
	const data = useAwaited(props);
	return props.children(data);
}
//#endregion
export { Await, useAwaited };

//# sourceMappingURL=awaited.js.map