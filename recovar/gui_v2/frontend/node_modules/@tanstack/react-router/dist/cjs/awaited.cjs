const require_runtime = require("./_virtual/_rolldown/runtime.cjs");
const require_utils = require("./utils.cjs");
let _tanstack_router_core = require("@tanstack/router-core");
let react = require("react");
react = require_runtime.__toESM(react);
let react_jsx_runtime = require("react/jsx-runtime");
//#region src/awaited.tsx
/** Suspend until a deferred promise resolves or rejects and return its data. */
function useAwaited({ promise: _promise }) {
	if (require_utils.reactUse) return require_utils.reactUse(_promise);
	const promise = (0, _tanstack_router_core.defer)(_promise);
	if (promise[_tanstack_router_core.TSR_DEFERRED_PROMISE].status === "pending") throw promise;
	if (promise[_tanstack_router_core.TSR_DEFERRED_PROMISE].status === "error") throw promise[_tanstack_router_core.TSR_DEFERRED_PROMISE].error;
	return promise[_tanstack_router_core.TSR_DEFERRED_PROMISE].data;
}
/**
* Component that suspends on a deferred promise and renders its child with
* the resolved value. Optionally provides a Suspense fallback.
*/
function Await(props) {
	const inner = /* @__PURE__ */ (0, react_jsx_runtime.jsx)(AwaitInner, { ...props });
	if (props.fallback) return /* @__PURE__ */ (0, react_jsx_runtime.jsx)(react.Suspense, {
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
exports.Await = Await;
exports.useAwaited = useAwaited;

//# sourceMappingURL=awaited.cjs.map