//#region src/Matches.ts
/**
* Narrows matches based on a path
* @experimental
*/
var isMatch = (match, path) => {
	const parts = path.split(".");
	let part;
	let i = 0;
	let value = match;
	while ((part = parts[i++]) != null && value != null) value = value[part];
	return value != null;
};
//#endregion
exports.isMatch = isMatch;

//# sourceMappingURL=Matches.cjs.map