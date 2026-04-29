import { ShallowErrorPlugin } from "./ShallowErrorPlugin.js";
import { RawStreamSSRPlugin } from "./RawStream.js";
import { ReadableStreamPlugin } from "seroval-plugins/web";
//#region src/ssr/serializer/seroval-plugins.ts
var defaultSerovalPlugins = [
	ShallowErrorPlugin,
	RawStreamSSRPlugin,
	ReadableStreamPlugin
];
//#endregion
export { defaultSerovalPlugins };

//# sourceMappingURL=seroval-plugins.js.map