const require_router = require("./router.cjs");
//#region src/defer.ts
/**
* Well-known symbol used by {@link defer} to tag a promise with
* its deferred state. Consumers can read `promise[TSR_DEFERRED_PROMISE]`
* to access `status`, `data`, or `error`.
*/
var TSR_DEFERRED_PROMISE = Symbol.for("TSR_DEFERRED_PROMISE");
/**
* Wrap a promise with a deferred state for use with `<Await>` and `useAwaited`.
*
* The returned promise is augmented with internal state (status/data/error)
* so UI can read progress or suspend until it settles.
*
* @param _promise The promise to wrap.
* @param options Optional config. Provide `serializeError` to customize how
* errors are serialized for transfer.
* @returns The same promise with attached deferred metadata.
* @link https://tanstack.com/router/latest/docs/framework/react/api/router/deferFunction
*/
function defer(_promise, options) {
	const promise = _promise;
	if (promise[TSR_DEFERRED_PROMISE]) return promise;
	promise[TSR_DEFERRED_PROMISE] = { status: "pending" };
	promise.then((data) => {
		promise[TSR_DEFERRED_PROMISE].status = "success";
		promise[TSR_DEFERRED_PROMISE].data = data;
	}).catch((error) => {
		promise[TSR_DEFERRED_PROMISE].status = "error";
		promise[TSR_DEFERRED_PROMISE].error = {
			data: (options?.serializeError ?? require_router.defaultSerializeError)(error),
			__isServerError: true
		};
	});
	return promise;
}
//#endregion
exports.TSR_DEFERRED_PROMISE = TSR_DEFERRED_PROMISE;
exports.defer = defer;

//# sourceMappingURL=defer.cjs.map