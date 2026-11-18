// NestQuant server desktop widget: a small window on the desktop that shows the server's live metrics
// (GET /metrics) every 2 seconds. Build with macos/widget/build.sh.
//
// Settings (defaults write ai.nestquant.widget <key> <value>):
//   url     metrics URL (default http://127.0.0.1:8000/metrics)
//   apiKey  API key if the server requires one
// Right-click the widget for "Keep on top", "Open metrics in browser" and "Quit". Drag it to move it.

import AppKit
import IOKit.ps

let defaults = UserDefaults.standard      // the app's own domain, ai.nestquant.widget

// MARK: - formatting

func compact(_ value: Double?) -> String {
    guard let v = value else { return "–" }
    if v >= 1_000_000 { return String(format: "%.2fM", v / 1_000_000) }
    if v >= 10_000 { return String(format: "%.0fK", v / 1_000) }
    if v >= 1_000 { return String(format: "%.1fK", v / 1_000) }
    return String(format: "%.0f", v)
}

func rate(_ value: Double?) -> String {
    guard let v = value else { return "–" }
    return String(format: "%.1f", v)
}

func duration(_ seconds: Double?) -> String {
    let s = Int(seconds ?? 0)
    if s >= 3600 { return "\(s / 3600)h\(String(format: "%02d", s % 3600 / 60))m" }
    if s >= 60 { return "\(s / 60)m\(String(format: "%02d", s % 60))s" }
    return "\(s)s"
}

func num(_ any: Any?) -> Double? {
    if let n = any as? NSNumber { return n.doubleValue }
    return nil
}

func dict(_ any: Any?) -> [String: Any] { any as? [String: Any] ?? [:] }

// MARK: - battery

func batteryText() -> String? {
    guard let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
          let list = IOPSCopyPowerSourcesList(info)?.takeRetainedValue() as? [CFTypeRef] else { return nil }
    for source in list {
        guard let d = IOPSGetPowerSourceDescription(info, source)?.takeUnretainedValue() as? [String: Any],
              let capacity = d[kIOPSCurrentCapacityKey] as? Int else { continue }
        let onAC = (d[kIOPSPowerSourceStateKey] as? String) == kIOPSACPowerValue
        let charging = d[kIOPSIsChargingKey] as? Bool ?? false
        return "battery \(capacity)%" + (charging ? " ⚡" : (onAC ? " (AC)" : " (on battery)"))
    }
    return nil
}

// MARK: - view

final class Widget: NSObject, NSApplicationDelegate {
    var panel: NSPanel!
    let label = NSTextField(labelWithString: "")
    var timer: Timer?

    let mono = NSFont.monospacedDigitSystemFont(ofSize: 12, weight: .regular)
    let monoBold = NSFont.monospacedDigitSystemFont(ofSize: 12, weight: .semibold)
    let head = NSFont.systemFont(ofSize: 10, weight: .semibold)

    func applicationDidFinishLaunching(_ notification: Notification) {
        panel = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 480, height: 250),
                        styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.isMovableByWindowBackground = true
        panel.collectionBehavior = [.canJoinAllSpaces, .stationary, .ignoresCycle]
        panel.hidesOnDeactivate = false
        applyLevel()

        let effect = NSVisualEffectView()
        effect.material = .hudWindow
        effect.state = .active
        effect.blendingMode = .behindWindow
        effect.wantsLayer = true
        effect.layer?.cornerRadius = 14
        effect.layer?.masksToBounds = true
        panel.contentView = effect

        label.maximumNumberOfLines = 0
        label.lineBreakMode = .byWordWrapping
        label.translatesAutoresizingMaskIntoConstraints = false
        label.preferredMaxLayoutWidth = 450
        effect.addSubview(label)
        NSLayoutConstraint.activate([
            label.leadingAnchor.constraint(equalTo: effect.leadingAnchor, constant: 14),
            label.trailingAnchor.constraint(equalTo: effect.trailingAnchor, constant: -14),
            label.topAnchor.constraint(equalTo: effect.topAnchor, constant: 12),
            label.bottomAnchor.constraint(equalTo: effect.bottomAnchor, constant: -12),
        ])

        let menu = NSMenu()
        let top = NSMenuItem(title: "Keep on top", action: #selector(toggleTop), keyEquivalent: "")
        top.target = self
        top.state = defaults.bool(forKey: "keepOnTop") ? .on : .off
        menu.addItem(top)
        let open = NSMenuItem(title: "Open metrics in browser", action: #selector(openMetrics), keyEquivalent: "")
        open.target = self
        menu.addItem(open)
        menu.addItem(.separator())
        let quit = NSMenuItem(title: "Quit NQ Widget", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "")
        menu.addItem(quit)
        effect.menu = menu

        panel.setFrameAutosaveName("NQWidgetFrame")
        if !panel.setFrameUsingName("NQWidgetFrame"), let screen = NSScreen.main {
            let f = screen.visibleFrame
            panel.setFrameOrigin(NSPoint(x: f.maxX - 500, y: f.maxY - 280))
        }
        render(nil, error: "connecting…")
        if let index = CommandLine.arguments.firstIndex(of: "--snapshot"), index + 1 < CommandLine.arguments.count {
            snapshot(to: CommandLine.arguments[index + 1])
            return
        }
        panel.orderFrontRegardless()

        timer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in self?.poll() }
        poll()
    }

    /// Render one update into a PNG and quit (for checking the layout without a screen capture).
    func snapshot(to path: String) {
        var request = URLRequest(url: metricsURL, timeoutInterval: 4)
        if let key = defaults.string(forKey: "apiKey"), !key.isEmpty {
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        }
        URLSession.shared.dataTask(with: request) { [weak self] data, _, error in
            let object = data.flatMap { try? JSONSerialization.jsonObject(with: $0) } as? [String: Any]
            DispatchQueue.main.async {
                guard let self = self, let view = self.panel.contentView else { exit(1) }
                self.render(object, error: error?.localizedDescription ?? (object == nil ? "no data" : nil))
                view.layoutSubtreeIfNeeded()
                view.appearance = NSAppearance(named: .darkAqua)
                if let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) {
                    view.cacheDisplay(in: view.bounds, to: rep)
                    try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: path))
                }
                exit(0)
            }
        }.resume()
    }

    func applyLevel() {
        if defaults.bool(forKey: "keepOnTop") {
            panel.level = .floating
        } else {
            // Just above the desktop icons, below normal windows, like a desktop widget.
            panel.level = NSWindow.Level(rawValue: Int(CGWindowLevelForKey(.desktopIconWindow)) + 1)
        }
    }

    @objc func toggleTop(_ sender: NSMenuItem) {
        let on = !defaults.bool(forKey: "keepOnTop")
        defaults.set(on, forKey: "keepOnTop")
        sender.state = on ? .on : .off
        applyLevel()
        panel.orderFrontRegardless()
    }

    var metricsURL: URL {
        URL(string: defaults.string(forKey: "url") ?? "http://127.0.0.1:8000/metrics")!
    }

    @objc func openMetrics() { NSWorkspace.shared.open(metricsURL) }

    func poll() {
        var request = URLRequest(url: metricsURL, timeoutInterval: 4)
        if let key = defaults.string(forKey: "apiKey"), !key.isEmpty {
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        }
        URLSession.shared.dataTask(with: request) { [weak self] data, _, error in
            let object = data.flatMap { try? JSONSerialization.jsonObject(with: $0) } as? [String: Any]
            DispatchQueue.main.async {
                if let m = object { self?.render(m, error: nil) }
                else { self?.render(nil, error: error?.localizedDescription ?? "no data") }
            }
        }.resume()
    }

    // MARK: rendering

    func render(_ m: [String: Any]?, error: String?) {
        let text = NSMutableAttributedString()
        let white = NSColor.white, grey = NSColor(white: 0.62, alpha: 1)

        func add(_ s: String, _ font: NSFont, _ color: NSColor) {
            text.append(NSAttributedString(string: s, attributes: [.font: font, .foregroundColor: color]))
        }
        func row(_ title: String, _ value: String, color: NSColor = .white) {
            add(title + "\t", head, grey)
            add(value + "\n", mono, color)
        }

        guard let m = m else {
            add("● ", monoBold, .systemRed)
            add("NestQuant server\n", NSFont.systemFont(ofSize: 13, weight: .semibold), white)
            add("not reachable: \(error ?? "")\n", mono, grey)
            add("start it with ~/startnq.sh\n", mono, grey)
            if let b = batteryText() { add(b, mono, grey) }
            label.attributedStringValue = text
            return
        }

        let requests = dict(m["requests"]), current = m["current"] as? [String: Any]
        let last = m["last"] as? [String: Any], recent = dict(m["recent"]), totals = dict(m["totals"])
        let gpu = dict(m["gpu"]), memory = dict(m["memory"]), cache = m["cache"] as? [String: Any]
        let running = num(requests["running"]) ?? 0, queued = num(requests["queued"]) ?? 0

        add("● ", monoBold, running > 0 ? .systemOrange : .systemGreen)
        let models = (m["models"] as? [String])?.joined(separator: ", ") ?? ""
        add("NestQuant  ", NSFont.systemFont(ofSize: 13, weight: .semibold), white)
        add("\(models) · up \(duration(num(m["uptime_seconds"])))\n", mono, grey)

        row("REQUESTS", "\(Int(running)) processing · \(Int(queued)) queued · \(Int(num(requests["served"]) ?? 0)) served"
            + ((num(requests["failed"]) ?? 0) > 0 ? " · \(Int(num(requests["failed"])!)) failed" : ""),
            color: queued > 0 ? .systemYellow : white)

        if let c = current {
            if (c["phase"] as? String) == "prefill" {
                row("NOW", "reading \(compact(num(c["prompt_tokens"])))-token prompt · \(duration(num(c["elapsed_seconds"])))",
                    color: .systemOrange)
            } else {
                row("NOW", "writing \(compact(num(c["generated_tokens"]))) tok @ \(rate(num(c["decode_tok_s"]))) tok/s · "
                    + "\(duration(num(c["elapsed_seconds"])))" + ((c["thinking"] as? String) != nil ? " · thinking" : ""),
                    color: .systemOrange)
            }
        } else {
            row("NOW", "idle", color: grey)
        }

        row("DECODE", "last \(rate(num(last?["decode_tok_s"]))) · avg \(rate(num(recent["decode_tok_s"]))) tok/s"
            + " · session \(rate(num(totals["decode_tok_s"])))")
        let lastNew = num(last?["prefilled_tokens"])
        row("PREFILL", "last \(compact(lastNew)) new in \(rate(num(last?["prefill_seconds"]))) s"
            + " · large avg \(rate(num(recent["large_prefill_tok_s"]))) tok/s")
        row("1ST TOKEN", "avg \(rate(num(recent["time_to_first_token_seconds"]))) s over last \(Int(num(recent["requests"]) ?? 0))")
        row("SESSION", "\(compact(num(totals["completion_tokens"]))) generated · \(compact(num(totals["prompt_tokens"]))) prompt"
            + " · \(rate(num(totals["cache_reuse_percent"])))% from cache")

        if let cache = cache {
            let sizes = (cache["cached_tokens"] as? [NSNumber])?.map { compact($0.doubleValue) }.joined(separator: ", ") ?? ""
            let misses = Int(num(cache["eviction_misses"]) ?? 0)
            row("CACHE", "\(Int(num(cache["cached"]) ?? 0))/\(Int(num(cache["slots"]) ?? 0)) conversations"
                + (sizes.isEmpty ? "" : " (\(sizes))") + " · \(misses) re-read after eviction"
                + ((cache["shared_starts"] as? [NSNumber]).flatMap { $0.isEmpty ? nil : " · shared start "
                    + $0.map { compact($0.doubleValue) }.joined(separator: ", ") } ?? ""),
                color: misses > 0 ? .systemYellow : white)
        }
        if let modes = recent["modes"] as? [String: NSNumber], !modes.isEmpty {
            let order = ["extend", "rollback", "replay", "fresh"]
            let parts = order.compactMap { k in modes[k].map { "\(k) \($0.intValue)" } }
            row("REUSE", parts.joined(separator: " · ") + "  (last 50)")
        }
        if let advice = m["advice"] as? String {
            let warn = advice.hasPrefix("Raise") || advice.contains("will not help")
            add("\t", mono, grey)
            add((warn ? "▲ " : "✓ ") + advice + "\n", mono, warn ? .systemYellow : .systemGreen)
        }

        var system = "GPU \(Int(num(gpu["utilization_percent"]) ?? 0))% · server \(rate(num(memory["process_gib"]))) GiB"
            + " · free \(rate(num(memory["system_available_gib"]))) GiB"
        if let pressure = memory["pressure"] as? String, pressure != "normal" { system += " (\(pressure))" }
        if let b = batteryText() { system += " · " + b }
        let low = (memory["pressure"] as? String).map { $0 != "normal" } ?? false
        row("SYSTEM", system, color: low ? .systemYellow : white)

        // drop the final newline; one tab stop aligns the values, and wrapped lines indent to it
        if text.string.hasSuffix("\n") { text.deleteCharacters(in: NSRange(location: text.length - 1, length: 1)) }
        let style = NSMutableParagraphStyle()
        style.tabStops = [NSTextTab(textAlignment: .left, location: 78)]
        style.headIndent = 78
        style.lineSpacing = 2
        text.addAttribute(.paragraphStyle, value: style, range: NSRange(location: 0, length: text.length))
        label.attributedStringValue = text
        let size = label.fittingSize
        let frame = panel.frame
        let height = size.height + 24
        if abs(frame.height - height) > 1 {
            panel.setFrame(NSRect(x: frame.minX, y: frame.maxY - height, width: 480, height: height), display: true)
        }
    }
}

let app = NSApplication.shared
let delegate = Widget()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
