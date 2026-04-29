//#region src/ssr/json.ts
/**
* @deprecated Use [`Response.json`](https://developer.mozilla.org/en-US/docs/Web/API/Response/json_static) from the standard Web API directly.
*/
function json(payload, init) {
	return Response.json(payload, init);
}
//#endregion
export { json };

//# sourceMappingURL=json.js.map