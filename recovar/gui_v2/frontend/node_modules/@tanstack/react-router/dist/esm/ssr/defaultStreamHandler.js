import { RouterServer } from "./RouterServer.js";
import { renderRouterToStream } from "./renderRouterToStream.js";
import { jsx } from "react/jsx-runtime";
import { defineHandlerCallback } from "@tanstack/router-core/ssr/server";
//#region src/ssr/defaultStreamHandler.tsx
var defaultStreamHandler = defineHandlerCallback(({ request, router, responseHeaders }) => renderRouterToStream({
	request,
	router,
	responseHeaders,
	children: /* @__PURE__ */ jsx(RouterServer, { router })
}));
//#endregion
export { defaultStreamHandler };

//# sourceMappingURL=defaultStreamHandler.js.map