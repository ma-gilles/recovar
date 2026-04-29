import { RouterServer } from "./RouterServer.js";
import { renderRouterToString } from "./renderRouterToString.js";
import { jsx } from "react/jsx-runtime";
import { defineHandlerCallback } from "@tanstack/router-core/ssr/server";
//#region src/ssr/defaultRenderHandler.tsx
var defaultRenderHandler = defineHandlerCallback(({ router, responseHeaders }) => renderRouterToString({
	router,
	responseHeaders,
	children: /* @__PURE__ */ jsx(RouterServer, { router })
}));
//#endregion
export { defaultRenderHandler };

//# sourceMappingURL=defaultRenderHandler.js.map