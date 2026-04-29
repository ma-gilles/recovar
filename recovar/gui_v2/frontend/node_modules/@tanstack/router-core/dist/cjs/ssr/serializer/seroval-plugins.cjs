const require_ShallowErrorPlugin = require("./ShallowErrorPlugin.cjs");
const require_RawStream = require("./RawStream.cjs");
let seroval_plugins_web = require("seroval-plugins/web");
//#region src/ssr/serializer/seroval-plugins.ts
var defaultSerovalPlugins = [
	require_ShallowErrorPlugin.ShallowErrorPlugin,
	require_RawStream.RawStreamSSRPlugin,
	seroval_plugins_web.ReadableStreamPlugin
];
//#endregion
exports.defaultSerovalPlugins = defaultSerovalPlugins;

//# sourceMappingURL=seroval-plugins.cjs.map