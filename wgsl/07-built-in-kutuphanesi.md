---
title: Built-in Kütüphanesi
section: "17-18"
source: "W3C WGSL Spec §17–§18"
---

# 7. Built-in Kütüphanesi

> Referans sözlüğü: WGSL'in 13 kategoride ~130+ yerleşik fonksiyonu ve formal grammar.

---

## §17 Built-in Functions

Belirli fonksiyonlar **predeclared** (önceden bildirilmiş) olarak implementasyon tarafından sağlanır ve bir WGSL modülünde her zaman kullanılabilir. Bunlara **built-in functions** (yerleşik fonksiyonlar) denir.

Bir built-in fonksiyon, aynı ada sahip bir fonksiyon **ailesidir**; ancak parametre sayısı, sırası ve tipleri ile birbirinden ayrılır. Bu farklı varyasyonların her biri bir **overload**'dur.

> **Not:** Her kullanıcı tanımlı fonksiyonun yalnızca **bir** overload'u vardır.

Her overload şu bilgilerle tanımlanır:
- **Type parameterizations** (tip parametreleri) — varsa
- **Fonksiyon imzası** — ad, parametreler ve dönüş tipi
- **Davranış** — overload'un çalışma mantığı

Built-in fonksiyon çağrılırken, tüm argümanlar fonksiyon değerlendirmesi başlamadan **önce** değerlendirilir (bkz. §11.2).

---

### 17.1 Constructor Built-in Functions

**Value constructor** built-in fonksiyonları, belirli bir tipin değerini açıkça oluşturur. WGSL, tüm predeclared tipler ve tüm constructible structure tipleri için value constructor sağlar.

Constructor fonksiyonun adı, tip adıyla (veya type alias ile) aynı yazılır.

> **Not:** `frexp`, `modf` ve `atomicCompareExchangeWeak` tarafından döndürülen structure tipleri WGSL modüllerinde yazılamaz.

WGSL iki tür value constructor sağlar:
1. **Zero value constructors** — sıfır değer oluşturma
2. **Value constructors** — değer oluşturma ve dönüştürme

#### 17.1.1 Zero Value Built-in Functions

Her concrete, constructible `T` tipinin benzersiz bir **zero value** (sıfır değeri) vardır. WGSL'de `T()` şeklinde yazılır.

| Tip | Zero Value | Açıklama |
|-----|-----------|----------|
| `bool()` | `false` | Boolean sıfır değeri |
| `i32()` | `0i` | İşaretli tam sayı sıfırı |
| `u32()` | `0u` | İşaretsiz tam sayı sıfırı |
| `f32()` | `0.0f` | 32-bit kayan nokta sıfırı |
| `f16()` | `0.0h` | 16-bit kayan nokta sıfırı |
| `vecN<T>()` | `N` adet `T` sıfırı | Vektör: her bileşen `T`'nin sıfır değeri |
| `matCxR<T>()` | `C×R` matris sıfırları | Matris: tüm elemanlar `T`'nin sıfır değeri |
| `array<E, N>()` | `N` adet `E` sıfırı | Dizi: her eleman `E`'nin sıfır değeri |
| `S()` | Sıfır üyeli struct | Structure: tüm üyeler sıfır değerli |

**Overload:**

```
@const @must_use fn T() -> T
```

- **Parameterization:** `T`, concrete constructible tip
- **Açıklama:** `T` tipinin sıfır değerini oluşturur.

> **Not:** AbstractInt'in sıfır değeri `0`, AbstractFloat'ın sıfır değeri `0.0`'dır; ancak bunlara erişmek için built-in fonksiyon yoktur.

> **Not:** WGSL'de atomic tipler, runtime-sized diziler veya constructible olmayan diğer tipler için zero built-in fonksiyon yoktur.

```wgsl
vec2<f32>()                 // İki f32 bileşenli sıfır vektör
vec2<f32>(0.0, 0.0)         // Aynı değer, açıkça yazılmış

vec3<i32>()                 // Üç i32 bileşenli sıfır vektör
vec3<i32>(0, 0, 0)          // Aynı değer, açıkça yazılmış

array<bool, 2>()            // İki boolean'lık sıfır dizi
array<bool, 2>(false, false) // Aynı değer, açıkça yazılmış
```

```wgsl
struct Student {
  grade: i32,
  GPA: f32,
  attendance: array<bool, 4>
}

fn func() {
  var s: Student;
  s = Student();                        // Student sıfır değeri
  s = Student(0, 0.0, array<bool, 4>(false, false, false, false)); // Açık form
  s = Student(i32(), f32(), array<bool, 4>());  // Sıfır üyelerle
}
```

#### 17.1.2 Value Constructor Built-in Functions

Value constructor built-in fonksiyonları, constructible bir değeri şu yollarla oluşturur:

1. **Kopyalama** — aynı tipte mevcut bir değerin kopyası (identity)
2. **Bileşenlerden oluşturma** — açık bileşen listesinden composite değer
3. **Dönüştürme** — başka bir değer tipinden dönüşüm

Vektör ve matris formları, bileşen tipini belirtmeden boyutları belirten overload'lar sağlar — bileşen tipi argümanlardan çıkarılır.

##### `array`

| Overload | Parameterization | Açıklama |
|----------|-----------------|----------|
| `@const @must_use fn array<T, N>(e1: T, ..., eN: T) -> array<T, N>` | `T` concrete constructible | Elemanlardan dizi oluşturma |
| `@const @must_use fn array(e1: T, ..., eN: T) -> array<T, N>` | `T` constructible | Tip çıkarımlı dizi oluşturma |

##### `bool`

| Overload | Parameterization | Açıklama |
|----------|-----------------|----------|
| `@const @must_use fn bool(e: T) -> bool` | `T` scalar | `T` bool ise identity; değilse boolean zorlama (`e` sıfır değer ise `false`, aksi halde `true`) |

##### `f16`

| Overload | Parameterization | Açıklama |
|----------|-----------------|----------|
| `@const @must_use fn f16(e: T) -> f16` | `T` scalar | `T` f16 ise identity; numeric scalar ise dönüşüm; bool ise `true`→`1.0h`, `false`→`0.0h` |

##### `f32`

| Overload | Parameterization | Açıklama |
|----------|-----------------|----------|
| `@const @must_use fn f32(e: T) -> f32` | `T` concrete scalar | `T` f32 ise identity; numeric scalar ise dönüşüm; bool ise `true`→`1.0f`, `false`→`0.0f` |

##### `i32`

| Overload | Parameterization | Açıklama |
|----------|-----------------|----------|
| `@const @must_use fn i32(e: T) -> i32` | `T` scalar | `T` i32 ise identity; u32 ise bit reinterpretation; float ise sıfıra doğru yuvarlama; bool ise `true`→`1i`, `false`→`0i` |

##### `u32`

| Overload | Parameterization | Açıklama |
|----------|-----------------|----------|
| `@const @must_use fn u32(e: T) -> u32` | `T` scalar | `T` u32 ise identity; i32 ise bit reinterpretation; float ise sıfıra doğru yuvarlama; bool ise `true`→`1u`, `false`→`0u` |

##### `mat2x2` · `mat2x3` · `mat2x4` · `mat3x2` · `mat3x3` · `mat3x4` · `mat4x2` · `mat4x3` · `mat4x4`

Her `matCxR` matris tipi için üç overload grubu vardır:

| Overload Türü | İmza | Açıklama |
|---------------|------|----------|
| **Dönüşüm** | `fn matCxR<T>(e: matCxR<S>) -> matCxR<T>` | `T≠S` ise floating point dönüşüm |
| **Sütun vektörlerinden** | `fn matCxR<T>(v1: vecR<T>, ..., vC: vecR<T>) -> matCxR<T>` | `C` adet sütun vektöründen oluşturma |
| **Elemanlardan** | `fn matCxR<T>(e1: T, ..., eN: T) -> matCxR<T>` | `C×R` adet elemandan oluşturma (column-major) |

- **`T`:** `AbstractFloat`, `f16` veya `f32`
- **Column-major sıralama:** Elemanlar sütun sütun doldurulur

```wgsl
// mat2x2 sütun vektörlerinden
let m = mat2x2<f32>(
  vec2<f32>(1.0, 2.0),   // 1. sütun
  vec2<f32>(3.0, 4.0)    // 2. sütun
);

// mat2x2 elemanlardan (aynı sonuç)
let m2 = mat2x2<f32>(1.0, 2.0, 3.0, 4.0);
```

##### `Structures`

Constructible structure tipi `S`, üyelerinin sırasıyla constructor argümanları olarak kullanıldığı bir constructor sağlar:

```wgsl
struct Particle {
  pos: vec3<f32>,
  vel: vec3<f32>,
  mass: f32
}

let p = Particle(vec3<f32>(0.0), vec3<f32>(1.0, 0.0, 0.0), 1.5);
```

##### `vec2` · `vec3` · `vec4`

Her `vecN` vektör tipi için çeşitli overload'lar:

| Overload Türü | İmza | Açıklama |
|---------------|------|----------|
| **Identity/Dönüşüm** | `fn vecN<T>(e: vecN<S>) -> vecN<T>` | Kopyalama veya component-wise dönüşüm |
| **Splat** | `fn vecN<T>(e: T) -> vecN<T>` | Tüm bileşenleri aynı değerle doldurma |
| **Bileşenlerden** | `fn vecN<T>(e1: T, ..., eN: T) -> vecN<T>` | Her bileşeni ayrı ayrı belirtme |
| **Alt vektörlerden** | `fn vec4<T>(e1: vec2<T>, e2: vec2<T>) -> vec4<T>` | Alt vektörlerden birleştirme |

```wgsl
let v1 = vec3<f32>(1.0, 2.0, 3.0);   // Bileşenlerden
let v2 = vec4<f32>(1.0);              // Splat: (1.0, 1.0, 1.0, 1.0)
let v3 = vec4<f32>(v1, 4.0);          // vec3 + skaler
let v4 = vec4<f32>(v1.xy, v1.yz);     // İki vec2'den
```

---

### 17.2 Bit Reinterpretation Built-in Functions

#### 17.2.1 `bitcast`

`bitcast<T>(e)`, bir değerin bit temsilini tip dönüşümü yapmadan yeniden yorumlar.

| Overload | Parameterization | Açıklama |
|----------|-----------------|----------|
| `@const @must_use fn bitcast<T>(e: S) -> T` | `T`, `S`: 32-bit numeric scalar/vector | Bitleri olduğu gibi `T` olarak yeniden yorumla |

**Geçerli bitcast kombinasyonları** (32-bit tipler arası):

| Kaynak → Hedef | `u32` | `i32` | `f32` |
|----------------|-------|-------|-------|
| `u32` | ✅ | ✅ | ✅ |
| `i32` | ✅ | ✅ | ✅ |
| `f32` | ✅ | ✅ | ✅ |

Vektör tipleri için component-wise: `bitcast<vec2<f32>>(vec2<u32>(…))` gibi.

**Özel durumlar:**
- `bitcast<f16>(vec2<f16>(…))` — iki f16'yı tek u32 gibi paketleme
- Kaynak veya hedef `f32` ise ve kaynak NaN veya ∞ bit deseni içeriyorsa, sonuç **indeterminate value** olabilir

```wgsl
// Float'ı unsigned integer olarak yorumla (bit manipülasyonu için)
let float_val = 1.0f;
let bits = bitcast<u32>(float_val);  // IEEE-754 bit deseni: 0x3F800000

// Integer'ı float olarak geri yorumla
let restored = bitcast<f32>(bits);   // 1.0f

// vec2<f16> ↔ u32 paketleme
let packed = bitcast<u32>(vec2<f16>(1.0h, 2.0h));
let unpacked = bitcast<vec2<f16>>(packed);
```

---

### 17.3 Logical Built-in Functions

#### 17.3.1 `all`

Bir boolean vektörün **tüm** bileşenlerinin `true` olup olmadığını test eder.

| Overload | Parameterization | Açıklama |
|----------|-----------------|----------|
| `@const @must_use fn all(e: bool) -> bool` | — | Identity (skaler) |
| `@const @must_use fn all(e: vecN<bool>) -> bool` | `N` ∈ {2, 3, 4} | Tüm bileşenler `true` ise `true` |

```wgsl
let v = vec3<bool>(true, true, false);
let result = all(v);     // false (üçüncü bileşen false)

let v2 = vec2<bool>(true, true);
let result2 = all(v2);   // true
```

#### 17.3.2 `any`

Bir boolean vektörün **en az bir** bileşeninin `true` olup olmadığını test eder.

| Overload | Parameterization | Açıklama |
|----------|-----------------|----------|
| `@const @must_use fn any(e: bool) -> bool` | — | Identity (skaler) |
| `@const @must_use fn any(e: vecN<bool>) -> bool` | `N` ∈ {2, 3, 4} | Herhangi bir bileşen `true` ise `true` |

```wgsl
let v = vec3<bool>(false, false, true);
let result = any(v);     // true

let v2 = vec2<bool>(false, false);
let result2 = any(v2);   // false
```

#### 17.3.3 `select`

Koşula bağlı olarak iki değerden birini seçer (component-wise ternary).

| Overload | Parameterization | Açıklama |
|----------|-----------------|----------|
| `@const @must_use fn select(f: T, t: T, cond: bool) -> T` | `T` scalar/vector | `cond` true ise `t`, değilse `f` |
| `@const @must_use fn select(f: vecN<T>, t: vecN<T>, cond: vecN<bool>) -> vecN<T>` | `T` scalar | Component-wise seçim |

> ⚠️ Parametre sırası `(false_value, true_value, condition)` — sezgisel olmayan sıralama!

```wgsl
let a = 10;
let b = 20;
let result = select(a, b, true);     // 20 (cond true → t seçilir)

// Component-wise seçim
let v1 = vec3<f32>(1.0, 2.0, 3.0);
let v2 = vec3<f32>(4.0, 5.0, 6.0);
let mask = vec3<bool>(true, false, true);
let result2 = select(v1, v2, mask);  // vec3(4.0, 2.0, 6.0)
```

---

### 17.4 Array Built-in Functions

#### 17.4.1 `arrayLength`

Runtime-sized dizinin eleman sayısını döndürür.

| Overload | Parameterization | Açıklama |
|----------|-----------------|----------|
| `@must_use fn arrayLength(p: ptr<storage, array<E>, AM>) -> u32` | `E` herhangi eleman tipi, `AM` read veya read_write | Runtime-sized dizinin uzunluğu |

> **Not:** Bu fonksiyon yalnızca `storage` adres uzayındaki runtime-sized diziler için kullanılabilir. Derleme zamanı boyutu bilinen diziler için `arrayLength` gerekmez.

```wgsl
struct Data {
  count: u32,
  values: array<f32>    // Runtime-sized dizi
}
@group(0) @binding(0) var<storage> data: Data;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let len = arrayLength(&data.values);
  if gid.x < len {
    // values[gid.x] ile güvenle çalış
  }
}
```

---

### 17.5 Numeric Built-in Functions

Numerik built-in fonksiyonlar WGSL'in **en büyük kategorisidir** — 63 fonksiyon içerir. Aşağıda alt kategorilere ayrılmıştır.

> **Component-wise semantik:** `T` bir vektör tipiyse, fonksiyon her bileşene bağımsız olarak uygulanır. Örneğin `abs(vec3(-1, 2, -3))` = `vec3(1, 2, 3)`.

#### Trigonometri & Açı

| Fonksiyon | İmza | Açıklama |
|-----------|------|----------|
| `acos(e)` | `@const fn acos(T) -> T` | Arc-cosine. `e` ∈ [−1, 1], sonuç radyan [0, π] |
| `acosh(e)` | `@const fn acosh(T) -> T` | Hiperbolik arc-cosine. `e` ≥ 1 |
| `asin(e)` | `@const fn asin(T) -> T` | Arc-sine. `e` ∈ [−1, 1], sonuç [−π/2, π/2] |
| `asinh(e)` | `@const fn asinh(T) -> T` | Hiperbolik arc-sine |
| `atan(e)` | `@const fn atan(T) -> T` | Arc-tangent. Sonuç [−π/2, π/2] |
| `atanh(e)` | `@const fn atanh(T) -> T` | Hiperbolik arc-tangent. `e` ∈ (−1, 1) |
| `atan2(y, x)` | `@const fn atan2(T, T) -> T` | İki argümanlı arc-tangent. Sonuç (−π, π] |
| `cos(e)` | `@const fn cos(T) -> T` | Cosine (radyan cinsinden) |
| `cosh(e)` | `@const fn cosh(T) -> T` | Hiperbolik cosine |
| `degrees(e)` | `@const fn degrees(T) -> T` | Radyanı dereceye çevir: `e × 180/π` |
| `radians(e)` | `@const fn radians(T) -> T` | Dereceyi radyana çevir: `e × π/180` |
| `sin(e)` | `@const fn sin(T) -> T` | Sine (radyan cinsinden) |
| `sinh(e)` | `@const fn sinh(T) -> T` | Hiperbolik sine |
| `tan(e)` | `@const fn tan(T) -> T` | Tangent (radyan cinsinden) |
| `tanh(e)` | `@const fn tanh(T) -> T` | Hiperbolik tangent |

- `T`: `f32`, `f16`, `AbstractFloat` veya bunların `vecN` varyantları
- Tüm trig fonksiyonlar **component-wise** çalışır

#### Üstel & Logaritmik

| Fonksiyon | İmza | Açıklama |
|-----------|------|----------|
| `exp(e)` | `@const fn exp(T) -> T` | e^e (doğal üstel) |
| `exp2(e)` | `@const fn exp2(T) -> T` | 2^e |
| `log(e)` | `@const fn log(T) -> T` | ln(e). `e` > 0 |
| `log2(e)` | `@const fn log2(T) -> T` | log₂(e). `e` > 0 |
| `pow(base, exp)` | `@const fn pow(T, T) -> T` | base^exp. Inherited: `exp2(exp * log2(base))` |
| `sqrt(e)` | `@const fn sqrt(T) -> T` | √e. `e` ≥ 0. Inherited: `1.0/inverseSqrt(e)` |
| `inverseSqrt(e)` | `@const fn inverseSqrt(T) -> T` | 1/√e. `e` > 0 |

#### Yuvarlama & Kırpma

| Fonksiyon | İmza | Açıklama |
|-----------|------|----------|
| `ceil(e)` | `@const fn ceil(T) -> T` | En küçük tamsayı ≥ `e` (yukarı yuvarlama) |
| `floor(e)` | `@const fn floor(T) -> T` | En büyük tamsayı ≤ `e` (aşağı yuvarlama) |
| `round(e)` | `@const fn round(T) -> T` | En yakın tamsayıya yuvarlama (banker's rounding) |
| `trunc(e)` | `@const fn trunc(T) -> T` | Sıfıra doğru yuvarlama (kesirli kısmı at) |
| `fract(e)` | `@const fn fract(T) -> T` | Kesirli kısım: `e - floor(e)` |
| `saturate(e)` | `@const fn saturate(T) -> T` | `clamp(e, 0.0, 1.0)` — [0, 1] aralığına sıkıştır |
| `clamp(e, low, high)` | `@const fn clamp(T, T, T) -> T` | `max(low, min(high, e))` — [low, high] aralığına sıkıştır |
| `step(edge, x)` | `@const fn step(T, T) -> T` | `x < edge` ise `0.0`, aksi halde `1.0` |
| `smoothstep(low, high, x)` | `@const fn smoothstep(T, T, T) -> T` | Hermite interpolasyon: `t*t*(3-2*t)` burada `t = clamp((x-low)/(high-low), 0, 1)` |

> **Not:** `clamp` integer tipler için de çalışır (`i32`, `u32`, `AbstractInt`).

#### Aritmetik & İşaret

| Fonksiyon | İmza | Açıklama |
|-----------|------|----------|
| `abs(e)` | `@const fn abs(T) -> T` | Mutlak değer. Float ve integer tipler |
| `sign(e)` | `@const fn sign(T) -> T` | İşaret: `e > 0` → `1`, `e < 0` → `−1`, `e == 0` → `0` |
| `min(e1, e2)` | `@const fn min(T, T) -> T` | Küçük olan. Float ve integer |
| `max(e1, e2)` | `@const fn max(T, T) -> T` | Büyük olan. Float ve integer |
| `mix(e1, e2, e3)` | `@const fn mix(T, T, T) -> T` | Doğrusal interpolasyon: `e1*(1−e3) + e2*e3` |
| `fma(e1, e2, e3)` | `@const fn fma(T, T, T) -> T` | Fused multiply-add: `e1*e2 + e3` (inherited accuracy) |
| `modf(e)` | `@const fn modf(T) -> __modf_result` | Tam ve kesirli parçaya ayır. `fract` + `whole` struct döndürür |
| `frexp(e)` | `@const fn frexp(T) -> __frexp_result` | Significand ve exponent'e ayır. `fract` (∈ [0.5, 1)) + `exp` struct döndürür |
| `ldexp(e1, e2)` | `@const fn ldexp(T, I) -> T` | `e1 × 2^e2` — `frexp`'in tersi |
| `quantizeToF16(e)` | `@const fn quantizeToF16(f32) -> f32` | `e`'yi f16 hassasiyetine kuantize et |

```wgsl
// modf örneği
let result = modf(3.75);
// result.fract = 0.75, result.whole = 3.0

// frexp örneği
let result2 = frexp(6.5);
// result2.fract ≈ 0.8125, result2.exp = 3 (6.5 = 0.8125 × 2³)

// mix (lerp) örneği
let color = mix(vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), 0.5);
// color = vec3(0.5, 0.0, 0.5) — kırmızı ile mavi arası
```

#### Vektör & Matris

| Fonksiyon | İmza | Açıklama |
|-----------|------|----------|
| `cross(e1, e2)` | `fn cross(vec3<T>, vec3<T>) -> vec3<T>` | Çapraz çarpım (sadece vec3) |
| `dot(e1, e2)` | `fn dot(vecN<T>, vecN<T>) -> T` | Nokta çarpım (iç çarpım) |
| `dot4U8Packed(e1, e2)` | `fn dot4U8Packed(u32, u32) -> u32` | Paketlenmiş 4×u8 dot product |
| `dot4I8Packed(e1, e2)` | `fn dot4I8Packed(u32, u32) -> i32` | Paketlenmiş 4×i8 dot product |
| `distance(e1, e2)` | `fn distance(T, T) -> S` | Öklid mesafesi: `length(e1 - e2)` |
| `length(e)` | `fn length(T) -> S` | Vektör uzunluğu (Öklid normu) |
| `normalize(e)` | `fn normalize(vecN<T>) -> vecN<T>` | Birim vektör: `e / length(e)` |
| `faceForward(e1, e2, e3)` | `fn faceForward(vecN, vecN, vecN) -> vecN` | `dot(e2,e3) < 0` ise `e1`, aksi halde `-e1` |
| `reflect(e1, e2)` | `fn reflect(vecN, vecN) -> vecN` | Yansıma vektörü: `e1 - 2*dot(e2,e1)*e2` |
| `refract(e1, e2, e3)` | `fn refract(vecN, vecN, S) -> vecN` | Kırılma vektörü (Snell yasası) |
| `determinant(e)` | `fn determinant(matNxN<T>) -> T` | Kare matrisin determinantı |
| `transpose(e)` | `fn transpose(matCxR<T>) -> matRxC<T>` | Matris transpozu |

```wgsl
// Yüzey normali ile ışık yönü arasındaki açı (diffuse lighting)
let N = normalize(surface_normal);
let L = normalize(light_dir);
let NdotL = max(dot(N, L), 0.0);

// Yansıma vektörü (specular lighting)
let R = reflect(-L, N);
```

#### Bit Manipülasyonu

| Fonksiyon | İmza | Açıklama |
|-----------|------|----------|
| `countLeadingZeros(e)` | `@const fn countLeadingZeros(T) -> T` | MSB'den itibaren sıfır bit sayısı |
| `countOneBits(e)` | `@const fn countOneBits(T) -> T` | `1` olan bit sayısı (popcount) |
| `countTrailingZeros(e)` | `@const fn countTrailingZeros(T) -> T` | LSB'den itibaren sıfır bit sayısı |
| `extractBits(e, offset, count)` | `@const fn extractBits(T, u32, u32) -> T` | Bit alanı çıkarma (signed/unsigned) |
| `firstLeadingBit(e)` | `@const fn firstLeadingBit(T) -> T` | En anlamlı `1` (unsigned) veya işaret bitinden farklı ilk bit (signed). Bulunamazsa `0xFFFFFFFF` |
| `firstTrailingBit(e)` | `@const fn firstTrailingBit(T) -> T` | En az anlamlı `1` bitin konumu. Bulunamazsa `0xFFFFFFFF` |
| `insertBits(e, newbits, offset, count)` | `@const fn insertBits(T, T, u32, u32) -> T` | Bit alanı yerleştirme |
| `reverseBits(e)` | `@const fn reverseBits(T) -> T` | Bit sırasını tersine çevir |

- `T`: `i32`, `u32` veya bunların `vecN` varyantları

```wgsl
let x: u32 = 0x00FF0000u;
let leading = countLeadingZeros(x);    // 8
let ones = countOneBits(x);            // 8
let trailing = countTrailingZeros(x);  // 16

// Bit alanı çıkarma
let val: u32 = 0xABCD1234u;
let bits = extractBits(val, 8u, 8u);   // 8. bitten başlayarak 8 bit çıkar: 0x12

// firstLeadingBit — log2 yaklaşımı için kullanışlı
let msb = firstLeadingBit(1024u);      // 10 (2^10 = 1024)
```

### 17.6 Derivative Built-in Functions

> ⚠️ Derivative fonksiyonları **yalnızca `@fragment` stage**'de kullanılabilir. Bu fonksiyonlar, quad içindeki komşu invocation'ların değerlerini kullanarak yaklaşık kısmi türevler hesaplar.

Derivative fonksiyonları **uniform control flow** içinde çağrılmalıdır; aksi halde `derivative_uniformity` diagnostic'i tetiklenir.

| Fonksiyon | Açıklama |
|-----------|----------|
| `dpdx(e)` | X eksenindeki kısmi türev (coarse veya fine — implementasyona bağlı) |
| `dpdxCoarse(e)` | X eksenindeki kısmi türev — coarse (daha az doğru, daha hızlı) |
| `dpdxFine(e)` | X eksenindeki kısmi türev — fine (daha doğru) |
| `dpdy(e)` | Y eksenindeki kısmi türev |
| `dpdyCoarse(e)` | Y eksenindeki kısmi türev — coarse |
| `dpdyFine(e)` | Y eksenindeki kısmi türev — fine |
| `fwidth(e)` | Manhattan genişliği: `abs(dpdx(e)) + abs(dpdy(e))` |
| `fwidthCoarse(e)` | Manhattan genişliği — coarse |
| `fwidthFine(e)` | Manhattan genişliği — fine |

- **Parameterization:** `T`, `f32` veya `vecN<f32>`
- **Overload:** `@must_use fn dpdx(e: T) -> T` (ve diğerleri aynı kalıpta)
- **Accuracy:** Infinite ULP (sonlu hata sınırı **yoktur**)

```wgsl
@fragment
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  // UV koordinatlarının değişim hızını hesapla
  let ddx = dpdx(uv);
  let ddy = dpdy(uv);

  // Mipmap seviyesi seçimi için fwidth kullanımı
  let fw = fwidth(uv);

  return vec4<f32>(fw, 0.0, 1.0);
}
```

> **Coarse vs Fine:** Coarse varyant, quad'daki tek bir farkı kullanabilirken, fine varyant iki komşu farkın ortalamasını hesaplar. Performans açısından coarse tercih edilebilir; doğruluk açısından fine daha iyidir. Belirtilmemiş (`dpdx`/`dpdy`/`fwidth`) varyantlar implementasyonun tercihine bırakılır.

---

### 17.7 Texture Built-in Functions

Texture fonksiyonları, GPU texture birimlerini sorgulamak, örneklemek ve yazmak için kullanılır.

#### Sorgulama Fonksiyonları

| Fonksiyon | Açıklama |
|-----------|----------|
| `textureDimensions(t)` | Texture boyutları (genişlik, yükseklik, derinlik). Opsiyonel mipmap seviyesi alır. |
| `textureNumLayers(t)` | Array texture'daki katman sayısı |
| `textureNumLevels(t)` | Mipmap seviye sayısı |
| `textureNumSamples(t)` | Multisampled texture'daki örnek sayısı |

```wgsl
@group(0) @binding(0) var t: texture_2d<f32>;

@compute @workgroup_size(1)
fn main() {
  let dims = textureDimensions(t);       // vec2<u32>(width, height)
  let dims_mip = textureDimensions(t, 1); // Mip seviye 1'in boyutları
  let levels = textureNumLevels(t);       // Mipmap seviye sayısı
}
```

#### Yükleme Fonksiyonları

| Fonksiyon | Açıklama |
|-----------|----------|
| `textureLoad(t, coords, ...)` | Belirli texel'i doğrudan koordinatla oku (filtreleme yok). Opsiyonel: array index, mip level, sample index |

```wgsl
@group(0) @binding(0) var t: texture_2d<f32>;

@fragment
fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let texel = textureLoad(t, vec2<i32>(pos.xy), 0);  // (x, y, mip_level)
  return texel;
}
```

#### Örnekleme Fonksiyonları

| Fonksiyon | Uniformity | Açıklama |
|-----------|-----------|----------|
| `textureSample(t, s, coords, ...)` | ⚠️ Uniform | Filtrelenmiş örnekleme (implicit LOD, derivative gerektirir) |
| `textureSampleBias(t, s, coords, bias, ...)` | ⚠️ Uniform | Mipmap bias ile örnekleme |
| `textureSampleCompare(t, s, coords, depth, ...)` | ⚠️ Uniform | Depth karşılaştırma ile örnekleme (shadow mapping) |
| `textureSampleCompareLevel(t, s, coords, depth, ...)` | ✅ | Explicit LOD 0 ile depth karşılaştırma |
| `textureSampleGrad(t, s, coords, ddx, ddy, ...)` | ✅ | Explicit gradient ile örnekleme |
| `textureSampleLevel(t, s, coords, level, ...)` | ✅ | Explicit mipmap seviyesi ile örnekleme |
| `textureSampleBaseClampToEdge(t, s, coords)` | ✅ | LOD 0, kenar sıkıştırmalı örnekleme |

> ⚠️ `textureSample`, `textureSampleBias`, `textureSampleCompare` **implicit derivative** kullanır ve bu nedenle **uniform control flow** gerektirir. Non-uniform dallanma içinde kullanımı `derivative_uniformity` hatasına yol açar.

```wgsl
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

@fragment
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  // Implicit LOD (derivative hesaplar — uniform control flow gerektirir)
  let color = textureSample(t, s, uv);

  // Explicit LOD (non-uniform dallanmada güvenle kullanılabilir)
  let color2 = textureSampleLevel(t, s, uv, 0.0);

  // Shadow mapping
  // let shadow = textureSampleCompareLevel(shadow_tex, shadow_sampler, uv, depth, 0);

  return color;
}
```

#### Yazma Fonksiyonları

| Fonksiyon | Açıklama |
|-----------|----------|
| `textureStore(t, coords, value)` | Storage texture'a yazma. Opsiyonel array index. |

```wgsl
@group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let color = vec4<f32>(f32(gid.x) / 256.0, f32(gid.y) / 256.0, 0.5, 1.0);
  textureStore(output, vec2<i32>(gid.xy), color);
}
```

#### Gather Fonksiyonları

| Fonksiyon | Açıklama |
|-----------|----------|
| `textureGather(component, t, s, coords, ...)` | 2×2 texel bölgesinden tek bir bileşen toplama |
| `textureGatherCompare(t, s, coords, depth_ref, ...)` | Depth karşılaştırma ile gather |

`textureGather`, dört komşu texel'den belirtilen bileşeni (0=R, 1=G, 2=B, 3=A) `vec4` olarak döndürür — shadow mapping ve post-processing filtrelerinde verimlidir.

---

### 17.8 Atomic Built-in Functions

Atomic fonksiyonlar, **storage** veya **workgroup** adres uzayındaki `atomic<T>` tiplerinde (T: `i32` veya `u32`) bölünmez (thread-safe) okuma-yazma işlemleri gerçekleştirir.

#### Temel Operasyonlar

| Fonksiyon | Açıklama |
|-----------|----------|
| `atomicLoad(p)` | Atomik okuma — mevcut değeri döndürür |
| `atomicStore(p, v)` | Atomik yazma — değeri `v` olarak ayarla |

#### Read-Modify-Write (Oku-Değiştir-Yaz) Operasyonları

Tüm RMW fonksiyonları **önceki değeri** döndürür:

| Fonksiyon | İşlem | Açıklama |
|-----------|-------|----------|
| `atomicAdd(p, v)` | `*p += v` | Atomik toplama |
| `atomicSub(p, v)` | `*p -= v` | Atomik çıkarma |
| `atomicMax(p, v)` | `*p = max(*p, v)` | Atomik maksimum |
| `atomicMin(p, v)` | `*p = min(*p, v)` | Atomik minimum |
| `atomicAnd(p, v)` | `*p &= v` | Atomik bitwise AND |
| `atomicOr(p, v)` | `*p \|= v` | Atomik bitwise OR |
| `atomicXor(p, v)` | `*p ^= v` | Atomik bitwise XOR |

#### Exchange Operasyonları

| Fonksiyon | Açıklama |
|-----------|----------|
| `atomicExchange(p, v)` | Değeri `v` ile değiştir, önceki değeri döndür |
| `atomicCompareExchangeWeak(p, cmp, v)` | `*p == cmp` ise `v` ile değiştir. `__atomic_compare_exchange_result` struct döndürür: `.old_value` (önceki değer) ve `.exchanged` (değişim yapıldı mı) |

```wgsl
@group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
var<workgroup> local_sum: atomic<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lid: u32) {
  // Workgroup-local atomik toplama
  atomicAdd(&local_sum, 1u);

  workgroupBarrier();

  // Tek bir thread global sayacı günceller
  if lid == 0u {
    let local_count = atomicLoad(&local_sum);
    atomicAdd(&counter, local_count);
  }
}
```

```wgsl
// Compare-and-swap ile lock-free güncelleme
fn try_update(p: ptr<storage, atomic<u32>, read_write>, expected: u32, desired: u32) -> bool {
  let result = atomicCompareExchangeWeak(p, expected, desired);
  return result.exchanged;
}
```

---

### 17.9 Data Packing Built-in Functions

Packing fonksiyonları, birden fazla küçük değeri tek bir `u32`'ye sıkıştırır.

| Fonksiyon | Açıklama |
|-----------|----------|
| `pack4x8snorm(e: vec4<f32>) -> u32` | 4 float'u 4×i8 signed normalized olarak paketle (her biri [−1, 1] → [−127, 127]) |
| `pack4x8unorm(e: vec4<f32>) -> u32` | 4 float'u 4×u8 unsigned normalized olarak paketle (her biri [0, 1] → [0, 255]) |
| `pack4xI8(e: vec4<i32>) -> u32` | 4 i32'yi 4×i8 olarak paketle (alt 8 bit alınır) |
| `pack4xU8(e: vec4<u32>) -> u32` | 4 u32'yi 4×u8 olarak paketle (alt 8 bit alınır) |
| `pack4xI8Clamp(e: vec4<i32>) -> u32` | 4 i32'yi [−128, 127] aralığına clamp edip paketle |
| `pack4xU8Clamp(e: vec4<u32>) -> u32` | 4 u32'yi [0, 255] aralığına clamp edip paketle |
| `pack2x16snorm(e: vec2<f32>) -> u32` | 2 float'u 2×i16 signed normalized olarak paketle |
| `pack2x16unorm(e: vec2<f32>) -> u32` | 2 float'u 2×u16 unsigned normalized olarak paketle |
| `pack2x16float(e: vec2<f32>) -> u32` | 2 f32'yi 2×f16 olarak paketle |

```wgsl
// Normal vektörü tek u32'ye sıkıştır (vertex veri optimizasyonu)
let normal = vec4<f32>(0.5, 0.5, 0.707, 0.0);
let packed = pack4x8snorm(normal);

// İki float'u tek u32'ye paketle
let uv = vec2<f32>(0.75, 0.25);
let packed_uv = pack2x16unorm(uv);
```

---

### 17.10 Data Unpacking Built-in Functions

Unpacking fonksiyonları, paketlenmiş `u32` değerlerini geri çözer.

| Fonksiyon | Açıklama |
|-----------|----------|
| `unpack4x8snorm(e: u32) -> vec4<f32>` | 4×i8 → 4 float [−1, 1] |
| `unpack4x8unorm(e: u32) -> vec4<f32>` | 4×u8 → 4 float [0, 1] |
| `unpack4xI8(e: u32) -> vec4<i32>` | 4×i8 → 4 i32 (sign-extend) |
| `unpack4xU8(e: u32) -> vec4<u32>` | 4×u8 → 4 u32 (zero-extend) |
| `unpack2x16snorm(e: u32) -> vec2<f32>` | 2×i16 → 2 float [−1, 1] |
| `unpack2x16unorm(e: u32) -> vec2<f32>` | 2×u16 → 2 float [0, 1] |
| `unpack2x16float(e: u32) -> vec2<f32>` | 2×f16 → 2 f32 |

```wgsl
// Pack → unpack roundtrip
let original = vec4<f32>(0.1, 0.5, 0.9, 1.0);
let packed = pack4x8unorm(original);
let recovered = unpack4x8unorm(packed);
// recovered ≈ original (kuantizasyon hatası ile)
```

---

### 17.11 Synchronization Built-in Functions

Senkronizasyon fonksiyonları, bellek tutarlılığı ve yürütme sıralaması sağlar.

| Fonksiyon | Açıklama |
|-----------|----------|
| `storageBarrier()` | Storage buffer bellek operasyonlarını sıralar. Aynı workgroup'taki invocation'lar için görünürlük garantisi |
| `textureBarrier()` | Texture bellek operasyonlarını sıralar |
| `workgroupBarrier()` | Workgroup bellek operasyonlarını sıralar **ve** bir kontrol bariyeri oluşturur (tüm invocation'lar bu noktada buluşur) |
| `workgroupUniformLoad(p)` | Bir workgroup değişkenini uniform olarak yükler. Dahili olarak barrier + load + barrier uygular |

> ⚠️ **Tüm barrier fonksiyonları uniform control flow gerektirir.** Non-uniform dallanma içinde çağrılırsa koşulsuz **shader-creation error** üretilir.

```wgsl
var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lid: u32) {
  // Her thread kendi verisini yazar
  shared_data[lid] = compute_value(lid);

  // Tüm thread'lerin yazmasını bekle (kontrol + bellek bariyeri)
  workgroupBarrier();

  // Artık komşu thread'lerin yazdığı veri güvenle okunabilir
  let neighbor = shared_data[(lid + 1u) % 256u];
}
```

```wgsl
// workgroupUniformLoad: koşullu dallanma güvenli hale gelir
var<workgroup> flag: u32;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lid: u32) {
  if lid == 0u {
    flag = check_condition();
  }
  // flag'ı tüm invocation'lar için uniform olarak yükle
  let uniform_flag = workgroupUniformLoad(&flag);
  if uniform_flag != 0u {
    // Tüm invocation'lar aynı dalı izler — uniform control flow
    workgroupBarrier();
  }
}
```

---

### 17.12 Subgroup Built-in Functions

Subgroup fonksiyonları, aynı subgroup'taki invocation'lar arasında **SIMT** (Single Instruction Multiple Thread) stilinde verimli iletişim ve hesaplama sağlar. `subgroups` uzantısı gerektirir.

> **Not:** Subgroup boyutu [4, 128] aralığında ve her zaman 2'nin kuvvetidir.

#### Reduction Operasyonları

| Fonksiyon | Açıklama |
|-----------|----------|
| `subgroupAdd(e)` | Tüm active invocation'ların `e` değerlerinin toplamı |
| `subgroupExclusiveAdd(e)` | Exclusive prefix sum (kendi değeri hariç) |
| `subgroupInclusiveAdd(e)` | Inclusive prefix sum (kendi değeri dahil) |
| `subgroupMul(e)` | Tüm active invocation'ların `e` değerlerinin çarpımı |
| `subgroupExclusiveMul(e)` | Exclusive prefix product |
| `subgroupInclusiveMul(e)` | Inclusive prefix product |
| `subgroupMax(e)` | Maksimum değer |
| `subgroupMin(e)` | Minimum değer |
| `subgroupAnd(e)` | Bitwise AND |
| `subgroupOr(e)` | Bitwise OR |
| `subgroupXor(e)` | Bitwise XOR |

#### Voting & Ballot

| Fonksiyon | Açıklama |
|-----------|----------|
| `subgroupAll(e: bool) -> bool` | Tüm active invocation'lar `true` ise `true` |
| `subgroupAny(e: bool) -> bool` | Herhangi bir active invocation `true` ise `true` |
| `subgroupBallot(e: bool) -> vec4<u32>` | Her invocation için 1 bit — bit maskesi döndürür |
| `subgroupElect() -> bool` | Yalnızca en düşük ID'li active invocation için `true` |

#### İletişim

| Fonksiyon | Açıklama |
|-----------|----------|
| `subgroupBroadcast(e, id)` | `id` numaralı invocation'ın `e` değerini tüm invocation'lara yayınla |
| `subgroupBroadcastFirst(e)` | En düşük ID'li active invocation'ın değerini yayınla |
| `subgroupShuffle(e, id)` | `id` numaralı invocation'dan `e` değerini al |
| `subgroupShuffleDown(e, delta)` | `invocation_id + delta` konumundaki invocation'dan değer al |
| `subgroupShuffleUp(e, delta)` | `invocation_id - delta` konumundaki invocation'dan değer al |
| `subgroupShuffleXor(e, mask)` | `invocation_id ^ mask` konumundaki invocation'dan değer al |

```wgsl
enable subgroups;

@compute @workgroup_size(256)
fn main(@builtin(subgroup_invocation_id) sid: u32,
        @builtin(subgroup_size) sg_size: u32) {
  let my_value = compute_something(sid);

  // Subgroup-wide toplam (register düzeyinde, paylaşımlı bellek gerektirmez)
  let total = subgroupAdd(my_value);

  // Prefix sum (parallel scan)
  let prefix = subgroupExclusiveAdd(my_value);

  // Butterfly reduction için XOR shuffle
  let partner_val = subgroupShuffleXor(my_value, 1u);
}
```

---

### 17.13 Quad Built-in Functions

Quad fonksiyonları, bir quad (2×2 invocation grubu) içinde veri alışverişi sağlar. `subgroups` uzantısı gerektirir.

| Fonksiyon | Açıklama |
|-----------|----------|
| `quadBroadcast(e, id)` | Quad içindeki `id` (0–3) numaralı invocation'ın değerini al |
| `quadSwapDiagonal(e)` | Çapraz komşuyla değer değiştir (0↔3, 1↔2) |
| `quadSwapX(e)` | Yatay komşuyla değer değiştir (0↔1, 2↔3) |
| `quadSwapY(e)` | Dikey komşuyla değer değiştir (0↔2, 1↔3) |

**Quad düzeni:**

```
┌───┬───┐
│ 0 │ 1 │   quadSwapX: 0↔1, 2↔3
├───┼───┤   quadSwapY: 0↔2, 1↔3
│ 2 │ 3 │   quadSwapDiagonal: 0↔3, 1↔2
└───┴───┘
```

```wgsl
enable subgroups;

@fragment
fn main(@location(0) value: f32) -> @location(0) vec4<f32> {
  // Komşu fragment'ın değerini al (X ekseninde)
  let neighbor_x = quadSwapX(value);

  // Manuel derivative hesaplaması
  let dfdx = neighbor_x - value;

  return vec4<f32>(dfdx, 0.0, 0.0, 1.0);
}
```

---

## §18 Grammar for Recursive Descent Parsing

Bu bölüm **normatif değildir** (non-normative). WGSL grammar'ı LALR(1) parser'a uygun formda belirtilmiştir; ancak bir implementasyon **recursive-descent parser** kullanabilir.

Normatif grammar doğrudan recursive-descent parser'da kullanılamaz çünkü bazı kuralları **sol-özyinelemeli** (left-recursive) dir. Aşağıdaki dönüşümler uygulanmıştır:

1. **Doğrudan ve dolaylı sol-özyineleme** ortadan kaldırılmıştır
2. **Boş üretimler** (epsilon kuralları) engellenmiştir
3. **Ortak ön ekler** kardeş üretimler arasında birleştirilmiştir

> **Not:** Sonuç grammar LL(1) **değildir**. Bazı nonterminal'lerde birden fazla üretim aynı lookahead kümesine sahiptir.

### Temel Grammar Kuralları

```bnf
translation_unit :
  | global_directive* global_decl* global_assert*
```

```bnf
global_directive :
  | 'diagnostic' '(' ident_pattern_token ',' diagnostic_rule_name ','? ')' ';'
  | 'enable' ident_pattern_token (',' ident_pattern_token)* ','? ';'
  | 'requires' ident_pattern_token (',' ident_pattern_token)* ','? ';'
```

```bnf
global_decl :
  | attribute* 'fn' ident '(' param_list? ')' ('->' attribute* type_specifier)? compound_statement
  | attribute* 'var' template_args? optionally_typed_ident ('=' expression)? ';'
  | global_value_decl ';'
  | 'alias' ident '=' type_specifier ';'
  | 'struct' ident '{' struct_member (',' struct_member)* ','? '}'
```

### İfade Grammar'ı

```bnf
expression :
  | unary_expression bitwise_expression.post.unary_expression
  | unary_expression relational_expression.post.unary_expression
  | unary_expression relational_expression.post.unary_expression '&&' ...
  | unary_expression relational_expression.post.unary_expression '||' ...
```

```bnf
unary_expression :
  | primary_expression component_or_swizzle_specifier?
  | '!' unary_expression
  | '&' unary_expression
  | '*' unary_expression
  | '-' unary_expression
  | '~' unary_expression
```

```bnf
primary_expression :
  | ident template_elaborated_ident.post.ident
  | ident template_elaborated_ident.post.ident argument_expression_list
  | literal
  | '(' expression ')'
```

### Statement Grammar'ı

```bnf
statement :
  | attribute* 'for' '(' for_init? ';' expression? ';' for_update? ')' compound_statement
  | attribute* 'if' expression compound_statement ('else' 'if' expression compound_statement)* ('else' compound_statement)?
  | attribute* 'loop' attribute* '{' statement* ('continuing' attribute* '{' statement* ('break' 'if' expression ';')? '}')? '}'
  | attribute* 'switch' expression attribute* '{' switch_clause+ '}'
  | attribute* 'while' expression compound_statement
  | attribute* '{' statement* '}'
  | 'break' ';'
  | 'const_assert' expression ';'
  | 'continue' ';'
  | 'discard' ';'
  | 'return' expression? ';'
  | variable_or_value_statement ';'
  | variable_updating_statement ';'
  | ident func_call_statement.post.ident ';'
```

### Operatör Öncelik Kuralları

```bnf
shift_expression.post.unary_expression :
  | (multiplicative_operator unary_expression)* (additive_operator unary_expression (multiplicative_operator unary_expression)*)*
  | shift_left unary_expression
  | shift_right unary_expression

multiplicative_operator : | '%' | '*' | '/'
additive_operator : | '+' | '-'

compound_assignment_operator :
  | shift_left_assign | shift_right_assign
  | '%=' | '&=' | '*=' | '+=' | '-=' | '/=' | '^=' | '|='
```

### Literal Grammar'ı

```bnf
literal : | bool_literal | float_literal | int_literal

bool_literal : | 'false' | 'true'

int_literal : | decimal_int_literal | hex_int_literal
decimal_int_literal : | /0[iu]?/ | /[1-9][0-9]*[iu]?/

float_literal : | decimal_float_literal | hex_float_literal
decimal_float_literal :
  | /0[fh]/
  | /[0-9]*\.[0-9]+([eE][+-]?[0-9]+)?[fh]?/
  | /[0-9]+[eE][+-]?[0-9]+[fh]?/
  | /[0-9]+\.[0-9]*([eE][+-]?[0-9]+)?[fh]?/
  | /[1-9][0-9]*[fh]/
```

> **Not:** Kısa formatlı token tanımları (regex'ler) spec'in ana bölümünden alınmalıdır.

---

> **Önceki:** [← Paralel Çalışma ve Doğruluk](06-paralel-calisma-ve-dogruluk.md) · **Ana Sayfa:** [README →](README.md)
