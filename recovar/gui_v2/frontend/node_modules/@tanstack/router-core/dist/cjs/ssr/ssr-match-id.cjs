//#region src/ssr/ssr-match-id.ts
function dehydrateSsrMatchId(id) {
	return id.replaceAll("/", "\0");
}
function hydrateSsrMatchId(id) {
	return id.replaceAll("\0", "/").replaceAll("�", "/");
}
//#endregion
exports.dehydrateSsrMatchId = dehydrateSsrMatchId;
exports.hydrateSsrMatchId = hydrateSsrMatchId;

//# sourceMappingURL=ssr-match-id.cjs.map