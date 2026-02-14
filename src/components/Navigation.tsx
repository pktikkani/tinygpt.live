"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Navigation() {
  const pathname = usePathname();

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-surface-border bg-background/90 backdrop-blur-sm">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-6">
        <Link href="/" className="flex items-center gap-2">
          <span className="text-amber glow-amber text-lg font-bold tracking-wider">
            NEURAL FORGE
          </span>
          <span className="text-muted text-xs">v0.1</span>
        </Link>

        <div className="flex items-center gap-6">
          <Link
            href="/"
            className={`text-sm transition-colors ${
              pathname === "/"
                ? "text-amber glow-amber"
                : "text-muted hover:text-foreground"
            }`}
          >
            [playground]
          </Link>
          <Link
            href="/help"
            className={`text-sm transition-colors ${
              pathname === "/help"
                ? "text-amber glow-amber"
                : "text-muted hover:text-foreground"
            }`}
          >
            [help]
          </Link>
        </div>
      </div>
    </nav>
  );
}
