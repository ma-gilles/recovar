//#region src/ssr/ssr-match-id.ts
function dehydrateSsrMatchId(id) {
	return id.replaceAll("/", "\0");
}
function hydrateSsrMatchId(id) {
	return id.replaceAll("\0", "/").replaceAll("�", "/");
}
//#endregion
export { dehydrateSsrMatchId, hydrateSsrMatchId };

//# sourceMappingURL=ssr-match-id.js.map