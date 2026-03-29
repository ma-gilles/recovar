import { Outlet } from "@tanstack/react-router";
import { Sidebar } from "../components/sidebar/Sidebar";

export function RootLayout(): React.JSX.Element {
  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-50">
      <Sidebar />
      <main className="flex-1 overflow-auto p-6">
        <div className="mx-auto max-w-[1400px]">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
