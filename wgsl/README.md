---
title: WGSL Spec — Türkçe Referans
version: "1.0"
date: 2026-02-14
---

# WGSL Spec — Türkçe Referans

> W3C WebGPU Shading Language spesifikasyonunun (§1–§18) modüler Türkçe çevirisi ve referans dokümanı.
>
> **Durum:** ✅ Tamamlandı — 7 dosya, ~6.000 satır, §1'den §18'e tüm normatif içerik.

## 📖 Doküman Haritası

| # | Dosya | Kapsam (TOC) | Açıklama |
|---|-------|-------------|----------|
| 1 | [Temeller ve Yapı](01-temeller-ve-yapi.md) | §1 – §5, §16 | Intro, WGSL Module, Textual Structure, Directives, Declaration/Scope, Keyword/Token Summary |
| 2 | [Tip Sistemi](02-tip-sistemi.md) | §6 | Type Checking, Plain Types, Enumerations, Memory Views, Textures/Samplers, Type Aliases |
| 3 | [Değişkenler ve İfadeler](03-degiskenler-ve-ifadeler.md) | §7 – §8 | var/let/const/override, Expressions (19 alt bölüm) |
| 4 | [Program Akışı ve Fonksiyonlar](04-program-akisi-ve-fonksiyonlar.md) | §9 – §11 | Statements, Control Flow, Behavior Analysis, Assertions, Functions, Alias Analysis |
| 5 | [GPU Arayüzü ve Bellek](05-gpu-arayuzu-ve-bellek.md) | §12 – §14 | 15 Attribute, Entry Points, Shader Interface, Built-in I/O, Memory Layout/Model |
| 6 | [Paralel Çalışma ve Doğruluk](06-paralel-calisma-ve-dogruluk.md) | §15 | Execution, Uniformity Analysis, Workgroups, Subgroups, Collective Ops, FP Evaluation |
| 7 | [Built-in Kütüphanesi](07-built-in-kutuphanesi.md) | §17 – §18 | 13 kategori built-in fonksiyon (~130+), Grammar for Recursive Descent |

## 🧭 Nasıl Kullanılır

- Her dosya bağımsız okunabilir şekilde tasarlanmıştır.
- Dosya sonlarındaki **Önceki / Sonraki** bağlantıları ile sıralı okuma yapılabilir.
- Frontmatter alanları VitePress / Docusaurus uyumludur.
- Kod örnekleri `` ```wgsl `` bloklarında, BNF gösterimleri `` ```bnf `` bloklarında verilmiştir.
- Karmaşık konular (memory layout, floating-point accuracy, uniformity vb.) tablo formatında sunulmuştur.

## 📋 Spec Bölüm Haritası (§1 → §18)

```
§1  Introduction ──────────────────────┐
§2  WGSL Module ───────────────────────┤
§3  Textual Structure ─────────────────┤── 01-temeller-ve-yapi.md
§4  Directives ────────────────────────┤
§5  Declaration and Scope ─────────────┤
§16 Keyword and Token Summary ─────────┘

§6  Types ─────────────────────────────── 02-tip-sistemi.md

§7  Variable and Value Declarations ───┐
§8  Expressions ───────────────────────┘── 03-degiskenler-ve-ifadeler.md

§9  Statements ────────────────────────┐
§10 Assertions ────────────────────────┤── 04-program-akisi-ve-fonksiyonlar.md
§11 Functions ─────────────────────────┘

§12 Attributes ────────────────────────┐
§13 Entry Points ──────────────────────┤── 05-gpu-arayuzu-ve-bellek.md
§14 Memory ────────────────────────────┘

§15 Execution ─────────────────────────── 06-paralel-calisma-ve-dogruluk.md

§17 Built-in Functions ────────────────┐
§18 Grammar for Recursive Descent ─────┘── 07-built-in-kutuphanesi.md
```

## 📋 Kaynak

- [W3C WGSL Specification](https://www.w3.org/TR/WGSL/)
- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
