---
title: Paralel Çalışma ve Doğruluk
section: 15
source: "W3C WGSL Spec §15"
---

# 6. Paralel Çalışma ve Doğruluk

> GPU'nun nasıl paralel çalıştığı, uniformity analizi, collective operasyonlar ve floating-point doğruluğu. Spec'in en "hardcore" teorik bölümü.

---

## §15 Execution

WGSL modülündeki bir shader, GPU üzerinde çok sayıda **invocation** (çağrı) olarak çalıştırılır. Bu bölüm, invocation'ların bireysel ve kolektif olarak nasıl yürütüldüğünü tanımlar.

---

### 15.1 Program Order Within an Invocation

Her bir invocation içinde, WGSL statement'ları **kontrol akışı sırasına** (control flow order) göre yürütülür.

- Bir statement, yürütme sırasında **sıfır veya daha fazla kez** çalıştırılabilir.
- Her bir yürütme, benzersiz bir **dynamic statement instance** oluşturur.
- İfade (expression) değerlendirme sırası **soldan sağa** (left-to-right):

```wgsl
// foo() her zaman bar()'dan ÖNCE değerlendirilir
let result = foo() + bar();
```

- İç içe geçmiş ifadeler **veri bağımlılıkları** (data dependencies) oluşturur — iç ifade, dış ifadeden önce değerlendirilmelidir.

---

### 15.2 Uniformity

**Uniformity** (tekdüzelik), GPU programlamanın en kritik kavramlarından biridir. Belirli operasyonlar — **collective operation**'lar (barrier, derivative, texture sampling) — tüm invocation'ların **eş zamanlı** olarak çalıştırmasını gerektirir.

#### Temel Kavramlar

- **Uniform control flow:** Aynı kapsamdaki tüm invocation'lar, programın aynı noktasını eş zamanlı olarak yürütür.
- **Non-uniform control flow:** İnvocation'ların bir alt kümesi operasyonu yürütürken diğerleri yürütmez. Bu durum, **yanlış veya taşınabilir olmayan** davranışa yol açar.

#### Non-uniform kaynakları

| Kaynak | Açıklama |
|--------|----------|
| Mutable modül kapsamı değişkenleri | `var<storage, read_write>` gibi yazılabilir global değişkenler |
| Çoğu built-in değer | `local_invocation_index`, `position` gibi invocation'a özgü değerler |
| User-defined input'lar | Vertex/Fragment shader girdileri |
| Belirli built-in fonksiyonlar | Non-uniform sonuç üreten fonksiyonlar |

#### Uniformity Failure

WGSL derleyicisi, her collective operasyonun uniform control flow içinde çalışıp çalışmadığını doğrulamak için **statik uniformity analizi** yapar. Doğrulama başarısız olursa:

| Durum | Sonuç |
|-------|-------|
| Derivative hesaplayan built-in (`textureSample`, `dpdx`, vb.) | `derivative_uniformity` diagnostic'i tetiklenir (varsayılan: `error`, `@diagnostic` ile değiştirilebilir) |
| Senkronizasyon built-in (`workgroupBarrier`, vb.) | Koşulsuz `error` → shader-creation error |
| Subgroup/Quad built-in | `subgroup_uniformity` diagnostic'i tetiklenir (varsayılan: `error`, `@diagnostic` ile değiştirilebilir) |

#### 15.2.1 Terminology and Concepts

Belirli bir invocation grubu için:

- **Uniform control flow:** Verilen **uniformity scope** (tekdüzelik kapsamı) içindeki tüm invocation'lar, programın aynı noktasını sanki adım adım (lockstep) yürütüyormuş gibi çalışır.
- **Uniformity scope'lar:**

| Kapsam | Açıklama |
|--------|----------|
| **Workgroup uniformity scope** | Compute shader'da aynı workgroup'taki tüm invocation'lar |
| **Draw uniformity scope** | Vertex/Fragment shader'da aynı draw command'deki tüm invocation'lar |
| **Subgroup uniformity scope** | `subgroup_uniformity` özelliği destekleniyorsa, aynı subgroup'taki tüm invocation'lar |

- **Uniform value:** Uniform control flow'da çalıştırılan ve tüm invocation'ların **aynı değeri** hesapladığı bir ifade.
- **Uniform variable:** Her canlı olduğu noktada tüm invocation'ların aynı değeri tuttuğu bir yerel değişken.

#### 15.2.2 Uniformity Analysis Overview

Uniformity analizi, her fonksiyonda iki şeyi doğrular:

1. Fonksiyon, çağırdığı diğer fonksiyonlar için uniformity gereksinimlerini karşılar.
2. Fonksiyon, her çağrıldığında uniformity gereksinimlerini karşılar.

**Analiz özellikleri:**

- **Sound (sağlam):** Uniformity gereksinimlerini ihlal eden bir program için her zaman hata tetiklenir.
- **Refactoring-safe:** Kodu fonksiyona çıkarmak veya fonksiyonu inline yapmak, geçerli bir shader'ı geçersiz kılmaz.
- **İzlenebilir:** Hata durumunda, root cause'a kadar uzanan bir implikasyon zinciri oluşturulabilir.

Analiz, çağrı grafiğinde **yaprak fonksiyonlardan entry point'e** doğru (topolojik sırada) ilerler. WGSL'de recursion yasak olduğu için döngü riski yoktur.

##### Potential-Trigger-Set

Her fonksiyon çağrısı için, çağrı non-uniform control flow'da yapılırsa tetiklenecek olan **triggering rule**'lar hesaplanır:

| Triggering Rule | Kapsam |
|----------------|--------|
| `derivative_uniformity` | Derivative hesaplayan fonksiyonlar |
| `subgroup_uniformity` | Subgroup/Quad built-in'ler |
| *(İsimsiz)* | Senkronizasyon fonksiyonları (filtrelenmesi mümkün değil) |

#### 15.2.3 Analyzing the Uniformity Requirements of a Function

Her fonksiyon iki aşamada analiz edilir:

1. **Graf oluşturma:** Fonksiyonun sözdizimi üzerinden yönlü bir graf kurulur.
2. **Graf keşfi:** Bu graf üzerinden kısıtlamalar hesaplanır ve olası uniformity failure tespit edilir.

##### Grafın Özel Düğümleri

| Düğüm | Anlam |
|--------|-------|
| `RequiredToBeUniform.error/warning/info` | "Bu noktaya ulaşan her şey uniform olmalıdır" (True benzeri) |
| `MayBeNonUniform` | "Bu şeyin uniformity'si garanti edilemez" (False benzeri) |
| `CF_start` | Fonksiyon başlangıcındaki kontrol akışı uniformity gereksinimi |
| `param_i` | Her formal parametre için bir düğüm |
| `Value_return` | Fonksiyonun dönüş değerinin uniformity'si |

**Temel kural:** `RequiredToBeUniform.error` → `MayBeNonUniform` arasında bir yol varsa, uniformity violation tetiklenir.

##### Fonksiyon Tag'leri

Her fonksiyon için hesaplanan tag'ler:

| Tag Türü | Değerler | Açıklama |
|----------|----------|----------|
| **Call site tag** | `CallSiteRequiredToBeUniform.S` / `CallSiteNoRestriction` | Çağrı noktasının uniform olması gerekip gerekmediği |
| **Function tag** | `ReturnValueMayBeNonUniform` / `NoRestriction` | Dönüş değerinin non-uniform olup olmadığı |
| **Parameter tag** | `ParameterRequiredToBeUniform.S` / `ParameterNoRestriction` | Parametre değerinin uniform olma gereksinimi |
| **Parameter return tag** | `ParameterReturnContentsRequiredToBeUniform` / `ParameterReturnNoRestriction` | Parametrenin dönüş değeri üzerindeki etkisi |
| **Pointer parameter tag** | `PointerParameterMayBeNonUniform` / `PointerParameterNoRestriction` | Pointer parametresinin çağrıdan sonraki etkisi |

#### 15.2.4 Pointer Desugaring

Uniformity analizi öncesinde, `function` adres uzayındaki pointer parametreler **yerel değişken bildirimi** olarak çözümlenir (desugar):

```wgsl
// Orijinal
fn foo(p: ptr<function, array<f32, 4>>, i: i32) -> f32 {
  let p1 = p;
  var x = i;
  let p2 = &((*p1)[x]);
  x = 0;
  *p2 = 5;
  return (*p1)[x];
}

// Analiz için eşdeğer form
fn foo_for_analysis(p: ptr<function, array<f32, 4>>, i: i32) -> f32 {
  var p_var = *p;             // p için yerel değişken
  let p1 = &p_var;            // p1 artık p_var'a referans
  var x = i;
  let x_tmp1 = x;             // x'in o anki değerini yakala
  let p2 = &(p_var[x_tmp1]);  // p1'in initializer'ını substitüe et
  x = 0;
  *(&(p_var[x_tmp1])) = 5;    // p2'nin initializer'ını substitüe et
  return (*(&p_var))[x];      // p1'in initializer'ını substitüe et
}
```

Bu dönüşüm, pointer'ın **root identifier**'ını her kullanım noktasında doğrudan ortaya çıkararak analizi basitleştirir.

#### 15.2.5 Function-scope Variable Value Analysis

Fonksiyon kapsamındaki değişkenlerin değeri, ona ulaşan **atamalar** (assignments) üzerinden analiz edilir.

| Atama Türü | Koşul |
|------------|-------|
| **Full assignment** | Skaler tip VEYA composite tipin her bileşenine değer atanmış |
| **Partial assignment** | Composite'ın yalnızca bir alt kümesine atama |

**Full/Partial reference kavramı:**

- **Full reference:** Bir değişkenin tüm bellek konumlarına erişim (`v` ifadesi)
- **Partial reference:** Yalnızca bir alt kümeye erişim (`v.member`, `arr[0]`)

> ⚠️ Partial reference üzerinden yapılan atama, değişkenin tüm konumlarını değiştirmemiş gibi ele alınır. Bu, analizin **konservatif** olmasına ve bazı programların gereksiz yere reddedilmesine neden olabilir.

#### 15.2.6 Uniformity Rules for Statements

Statement'lar için uniformity kuralları, gelen kontrol akışı düğümünü (`CF`) alır ve çıkan kontrol akışı düğümünü üretir. Temel kurallar:

| Statement | Sonuç CF |
|-----------|----------|
| Boş statement | `CF` (değişmez) |
| `{ s }` compound | `s`'nin analiz sonucu |
| `s1 s2` (sequential) | `s1` → `CF1`, `CF2` ← `s2(CF1)` |
| `if e s1 else s2` (behavior {Next}) | Koşul `e` değerlendirilir → `V`; dallar `V` ile analiz edilir; sonuç `CF` (divergence çözülür) |
| `if e s1 else s2` (diğer behavior) | `CFend` → {`CF1`, `CF2`} (divergence kalıcı) |
| `loop { s1 }` | İterasyon arası bağımlılık: `CF'` → `CF1` kenarı; behavior {Next} ise çıkış `CF` |
| `switch e { case: s_i }` | `if` ile benzer mantık |
| `return e;` | `Value_return` → `V(e)` kenarı oluşturulur |
| `e1 = e2;` | `LV` → `RV` kenarı (atanan değerin uniformity'si aktarılır) |

> **Not:** Behavior {Next} olan `if` ve `switch` statement'larında, kontrol akışı divergence'ı **çözülmüş** kabul edilir — sonuç CF, orijinal CF'ye döner.

#### 15.2.7 Uniformity Rules for Function Calls

Fonksiyon çağrıları, çağrılan fonksiyonun tag'lerine göre analiz edilir:

- `CallSiteRequiredToBeUniform.S` → Çağrı noktasının kontrol akışı `RequiredToBeUniform.S`'e bağlanır
- `ParameterRequiredToBeUniform.S` → Argümanın değer düğümü `RequiredToBeUniform.S`'e bağlanır
- `ReturnValueMayBeNonUniform` → Sonuç düğümü `MayBeNonUniform`'a bağlanır
- `ParameterReturnContentsRequiredToBeUniform` → Parametre, dönüş değerinin uniformity'sine etki eder

#### 15.2.8 Uniformity Rules for Expressions

İfade analizi, hem **kontrol akışı** (CF) hem de **değer** (V) düğümü üretir:

| İfade | V (değer) | CF çıkışı |
|-------|-----------|-----------|
| Literal / const | Uniform (kenarsız) | CF |
| Değişken okuma (let/const) | Başlangıç değerine bağlı | CF |
| Değişken okuma (var, function alanı) | Value analysis sonucuna bağlı | CF |
| Modül kapsamı var (mutable) | `MayBeNonUniform`'a kenar | CF |
| Modül kapsamı var (immutable / uniform) | Uniform (kenarsız) | CF |
| `e1 op e2` (binary) | `V` → {`V1`, `V2`} kenarlar | `CF2` |
| `e1 && e2` / `e1 \|\| e2` | Short-circuit: `V2`'nin CF'si `V1`'e bağlı | `CF2` |
| Fonksiyon çağrısı `f(args)` | §15.2.7 kuralları | `CF_after` |
| Built-in değer (non-uniform, ör: `position`) | `MayBeNonUniform` | CF |
| Built-in değer (uniform, ör: `workgroup_id`) | Uniform | CF |

#### 15.2.9 Annotating the Uniformity of Every Point

Analiz tamamlandığında, programın her noktasındaki uniformity durumu graftan çıkarılabilir:
- `RequiredToBeUniform.S` → `MayBeNonUniform` yolu: **uniformity failure**
- Yol üzerindeki düğümler: hata mesajındaki açıklama zinciri

#### 15.2.10 Examples

##### 15.2.10.1 Invalid `textureSample` Function Call

```wgsl
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

@fragment
fn main(@builtin(position) pos: vec4<f32>) {
  if pos.x > 1.0 {
    // ❌ GEÇERSİZ: textureSample derivative hesaplar;
    // pos non-uniform → if koşulu non-uniform → non-uniform control flow
    _ = textureSample(t, s, pos.xy);
  }
}
```

**Neden geçersiz?** `position` built-in değeri non-uniform → `pos.x > 1.0` koşulu non-uniform → `if` dalları non-uniform control flow oluşturur → `textureSample` (derivative hesaplar) non-uniform control flow'da çağrılır.

> **Çözüm:** `textureSample` yerine `textureSampleLevel` kullanmak (explicit LOD) derivative gerektirmez ve non-uniform control flow'da güvenle çağrılabilir.

##### 15.2.10.2 Function-scope Variable Uniformity

```wgsl
@group(0) @binding(0) var<storage, read_write> a: i32;
@group(0) @binding(1) var<uniform> b: i32;

@compute @workgroup_size(16, 1, 1)
fn main() {
  var x: i32;
  x = a;
  if x > 0 {
    // ❌ GEÇERSİZ: x, mutable modül-kapsamı 'a'dan türetildi
    workgroupBarrier();
  }
  x = b;
  if x < 0 {
    // ✅ GEÇERLİ: x artık immutable uniform 'b'den türetildi
    storageBarrier();
  }
}
```

Bu örnek, value analysis'in bir değişkenin ömrü boyunca **farklı uniformity dönemlerini** ayırt edebildiğini gösterir. İlk `if`'ten sonra `x` yeniden atandığı için, ikinci `if` bağımsız olarak analiz edilir.

##### 15.2.10.3 Composite Value Analysis Limitations

```wgsl
struct Inputs {
  @builtin(workgroup_id) wgid: vec3<u32>,           // Uniform
  @builtin(local_invocation_index) lid: u32   // Non-uniform
}

@compute @workgroup_size(16, 1, 1)
fn main(inputs: Inputs) {
  // ❌ GEÇERSİZ: inputs composite olarak non-uniform olarak işaretlenir
  // (lid non-uniform olduğu için tüm inputs non-uniform)
  if inputs.wgid.x == 1 {
    workgroupBarrier();
  }
}
```

**Çözüm:** Composite'ı ayırın:

```wgsl
@compute @workgroup_size(16, 1, 1)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
  // ✅ GEÇERLİ: wgid ayrı parametre → uniform olarak izlenir
  if wgid.x == 1 {
    workgroupBarrier();
  }
}
```

##### 15.2.10.4 Uniformity in a Loop

```wgsl
@compute @workgroup_size(16, 1, 1)
fn main(@builtin(local_invocation_index) lid: u32) {
  for (var i = 0u; i < 10; i++) {
    workgroupBarrier();  // ❌ GEÇERSİZ
    if (lid + i) > 7 {
      break;  // Non-uniform break → sonraki iterasyonlarda CF non-uniform
    }
  }
}
```

**Neden?** `lid` non-uniform → `break` koşulu non-uniform → bazı invocation'lar döngüden erken çıkar → sonraki iterasyonlarda `workgroupBarrier` non-uniform control flow'da çağrılır.

##### 15.2.10.5 User-defined Function Calls

```wgsl
fn scale(in1: f32, in2: f32) -> f32 {
  let v = in1 / in2;
  return v;
}

@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

@fragment
fn main(@builtin(position) pos: vec4<f32>) {
  let tmp = scale(pos.x, 0.5);
  if tmp > 1.0 {
    // ❌ GEÇERSİZ: scale'in dönüş değeri her iki parametreye bağlıdır
    // (ParameterReturnContentsRequiredToBeUniform);
    // pos.x non-uniform → tmp non-uniform → non-uniform control flow
    _ = textureSample(t, s, pos.xy);
  }
}
```

Analiz, `scale` fonksiyonunun her iki parametresini `ParameterReturnContentsRequiredToBeUniform` olarak etiketler — dönüş değerinin uniform olması için parametrelerin de uniform olması gerekir.

---

### 15.3 Compute Shaders and Workgroups

#### Workgroup Kavramı

**Workgroup**, bir compute shader entry point'ini eş zamanlı olarak çalıştıran ve `workgroup` adres uzayındaki değişkenleri paylaşan bir invocation kümesidir.

#### Workgroup Grid

Compute shader'ın **workgroup grid**'i, `@workgroup_size(x, y, z)` attribute'u ile tanımlanan 3D tamsayı koordinat uzayıdır:

- `0 ≤ i < workgroup_size_x`
- `0 ≤ j < workgroup_size_y`
- `0 ≤ k < workgroup_size_z`

Her grid noktası için **tam olarak bir invocation** bulunur.

#### Invocation ID'leri

| ID Türü | Tanım | Formül |
|---------|-------|--------|
| **Local invocation ID** | Workgroup içindeki `(i, j, k)` koordinatı | Doğrudan grid noktası |
| **Local invocation index** | Tek boyutlu doğrusal indeks | `i + (j × size_x) + (k × size_x × size_y)` |
| **Workgroup ID** | Dispatch grid'indeki workgroup koordinatı | `(⌊CSi / size_x⌋, ⌊CSj / size_y⌋, ⌊CSk / size_z⌋)` |
| **Global invocation ID** | Compute shader grid'indeki koordinat | `(CSi, CSj, CSk)` |

#### Dispatch

Dispatch komutu bir **dispatch size** `(group_count_x, group_count_y, group_count_z)` belirtir. Compute shader grid:

- `0 ≤ CSi < workgroup_size_x × group_count_x`
- `0 ≤ CSj < workgroup_size_y × group_count_y`
- `0 ≤ CSk < workgroup_size_z × group_count_z`

#### Garanti Verilmeyenler

WebGPU aşağıdaki garantileri **vermez**:

- ❌ Farklı workgroup'ların eş zamanlı çalışması
- ❌ Bir workgroup çalışırken diğerlerinin bloklanması
- ❌ Workgroup'ların belirli bir sırada başlatılması

> Bu, workgroup'lar arası senkronizasyonun **imkânsız** olduğu anlamına gelir — yalnızca aynı workgroup içindeki invocation'lar barrier'lar aracılığıyla senkronize edilebilir.

---

### 15.4 Fragment Shaders and Helper Invocations

#### Quad Kavramı

Fragment shader invocation'ları, komşu `position` değerlerine sahip **2×2 grid**'ler halinde organize edilir. Her grid bir **quad** olarak adlandırılır.

| Quad ID | Konum |
|---------|-------|
| 0 | Sol üst |
| 1 | Sağ üst |
| 2 | Sol alt |
| 3 | Sağ alt |

> **Not:** Quad ID için bir built-in değer erişimcisi yoktur.

#### Helper Invocations

Bir grafik primitive'in kenarında, quad'ı doldurmak için yeterli RasterizationPoint olmayabilir. Bu durumda GPU, boş pozisyonlar için **helper invocation**'lar oluşturur.

**Helper invocation kısıtlamaları:**

| Kısıtlama | Açıklama |
|-----------|----------|
| Yazma erişimi yok | `storage` veya `handle` adres uzaylarına yazma yapılamaz |
| Atomic sonuçları belirsiz | Atomic built-in fonksiyonları **indeterminate** (belirsiz) sonuç döndürür |
| Return değeri işlenmez | Entry point dönüş değeri pipeline'da ileriye aktarılmaz |

Helper invocation'lar yalnızca **derivative hesaplamaları** için yardımcı olarak bulunur.

> **Özel durum:** Bir quad'daki tüm invocation'lar helper invocation haline gelirse (ör: hepsi `discard` çalıştırdığında), quad'ın yürütmesi sonlandırılabilir — ancak bu, non-uniform control flow olarak **kabul edilmez**.

---

### 15.5 Subgroups

**Subgroup**, bir compute veya fragment shader'da eş zamanlı çalışan ve verimli veri paylaşımı yapabilen bir invocation kümesidir.

| Özellik | Açıklama |
|---------|----------|
| **Kapsam** | Her invocation tam olarak bir subgroup'a aittir |
| **Compute shader'da** | Her subgroup, belirli bir workgroup'un alt kümesidir |
| **Fragment shader'da** | Bir subgroup, birden fazla draw command'den invocation içerebilir |
| **Quad ilişkisi** | Her quad, tek bir subgroup içinde yer alır |

#### Subgroup Size

| Özellik | Değer |
|---------|-------|
| Erişim | `subgroup_size` built-in değeri |
| Aralık | [4, 128] — her zaman 2'nin kuvveti |
| Uniformity | Compute shader'da bir dispatch command içinde uniform; fragment shader'da draw command içinde uniform olması garanti değil |
| Cihaz aralığı | `[subgroupMinSize, subgroupMaxSize]` |

#### Subgroup ID'leri

| ID | Erişim | Aralık |
|----|--------|--------|
| `subgroup_invocation_id` | Subgroup içindeki benzersiz ID | `[0, subgroup_size - 1]` |
| `subgroup_id` | Workgroup içindeki subgroup ID (sadece compute) | `[0, num_subgroups - 1]` |

> ⚠️ `subgroup_invocation_id` ile `local_invocation_index` arasında **tanımlı bir ilişki yoktur**. Bu iki değer arasında eşleme varsaymak **taşınabilir olmayan** koda yol açar.

#### Divergence

Aynı subgroup'taki invocation'lar farklı kontrol akışı yolları izlediğinde, subgroup yürütmesinin **diverge** olduğu söylenir:

- **Active invocation:** Bir subgroup operasyonunu eş zamanlı olarak yürüten invocation'lar
- **Inactive invocation:** Aynı operasyonu yürütmeyen invocation'lar (ya farklı dalda ya da subgroup boyutunu aşan "hayalet" invocation'lar)
- Helper invocation'lar cihaza bağlı olarak active veya inactive olabilir.

---

### 15.6 Collective Operations

#### 15.6.1 Barriers

**Barrier**, bir programdaki bellek operasyonlarını sıralayan bir **senkronizasyon** fonksiyonudur.

**Control barrier**: Aynı workgroup'taki tüm invocation'lar tarafından eş zamanlı olarak yürütülür. Bu nedenle barrier'lar **yalnızca uniform control flow** içinde çağrılabilir.

```wgsl
@compute @workgroup_size(64)
fn main() {
  // Workgroup belleğine yazma
  shared_data[local_idx] = compute_value();

  // Tüm invocation'ların yazmasını bekle
  workgroupBarrier();

  // Artık diğer invocation'ların yazdığı değerler güvenle okunabilir
  let neighbor = shared_data[local_idx ^ 1];
}
```

WGSL'de üç barrier fonksiyonu vardır:

| Fonksiyon | Kapsamı |
|-----------|---------|
| `workgroupBarrier()` | Workgroup bellek ve yürütme senkronizasyonu |
| `storageBarrier()` | Storage buffer bellek sıralaması |
| `textureBarrier()` | Texture bellek sıralaması |

#### 15.6.2 Derivatives

**Partial derivative** (kısmi türev), bir değerin bir eksen boyunca değişim oranıdır. Fragment shader'daki invocation'lar aynı **quad** içinde işbirliği yaparak yaklaşık türevler hesaplar.

##### Derivative hesaplayan fonksiyonlar

**İmplicit derivative** (fragment coordinate üzerinden):
- `textureSample`, `textureSampleBias`, `textureSampleCompare`

**Explicit derivative** (kullanıcı belirtilen değer üzerinden):

| Fonksiyon Grubu | X Ekseni | Y Ekseni | Manhattan |
|----------------|----------|----------|-----------|
| **Coarse** | `dpdxCoarse` | `dpdyCoarse` | `fwidthCoarse` |
| **Fine** | `dpdxFine` | `dpdyFine` | `fwidthFine` |
| **Otomatik** | `dpdx` | `dpdy` | `fwidth` |

**Uniformity gereksinimi:** Derivative fonksiyonları **uniform control flow** içinde çağrılmalıdır. Aksi halde `derivative_uniformity` diagnostic'i tetiklenir ve sonuç **indeterminate value** olur.

> **Not:** Derivative'ler örtük bir **quad operasyonu** türüdür. Kullanımları `subgroups` uzantısını gerektirmez.

#### 15.6.3 Subgroup Operations

**Subgroup built-in fonksiyonları**, aynı subgroup'taki invocation'lar arasında verimli iletişim ve hesaplama sağlar. SIMT (Single Instruction Multiple Thread) operasyonlarıdır.

Active invocation'lar işbirliği yaparak sonuçları belirler. Taşınabilirlik, tüm invocation'ların active olduğu durumda (yani uniform control flow'da) maksimize edilir.

#### 15.6.4 Quad Operations

**Quad built-in fonksiyonları**, bir quad (2×2 invocation grubu) üzerinde çalışır. Quad içi veri iletişimi için kullanışlıdır.

Subgroup operasyonlarındakine benzer şekilde, taşınabilirlik uniform control flow'da maksimize edilir.

---

### 15.7 Floating Point Evaluation

WGSL'nin floating point özellikleri **IEEE-754** standardına dayanır; ancak GPU'ların yaptığı ödünleşimleri yansıtacak şekilde azaltılmış işlevsellik ve ek taşınabilirlik güvenceleri içerir.

#### 15.7.1 Overview of IEEE-754

IEEE-754 binary floating point tipi, **genişletilmiş gerçel sayı** (extended real number) doğrusunu yaklaştırır:

##### Değer Kategorileri

| Kategori | Açıklama |
|----------|----------|
| **Pozitif/Negatif rasyonel sayılar** | Sonlu; normal veya subnormal |
| **±∞ (Sonsuzluk)** | Pozitif ve negatif sonsuzluk |
| **NaN (Not a Number)** | Geçersiz operasyon sonucu |
| **±0 (Sıfır)** | Pozitif ve negatif sıfır (birbirine eşit) |

##### Bit Temsili

Her IEEE-754 floating point değeri üç bitfield'den oluşur (MSB → LSB):

| Alan | Genişlik |
|------|----------|
| **Sign (İşaret)** | 1 bit |
| **Exponent (Üs)** | Tipe bağlı |
| **Trailing significand (Anlamlı kısım)** | Tipe bağlı |

##### IEEE-754 Tipleri

| Tip | Exponent | Significand | Bias | Sonlu Aralık |
|-----|----------|-------------|------|-------------|
| **binary16** (f16) | 5 bit | 10 bit | 15 | [−65504, 65504] |
| **binary32** (f32) | 8 bit | 23 bit | 127 | ≈ [−3.4×10³⁸, 3.4×10³⁸] |
| **binary64** (AbstractFloat) | 11 bit | 52 bit | 1023 | ≈ [−1.8×10³⁰⁸, 1.8×10³⁰⁸] |

##### Değer Yorumlama Algoritması

Bit temsilinden değer:
- Exponent tüm 1'ler → T=0 ise ±∞, T≠0 ise NaN
- Exponent tüm 0'lar → (−1)^Sign × 2^(−bias) × T × 2^(−tsw+1) → T=0 ise sıfır, T≠0 ise **subnormal**
- Diğer → (−1)^Sign × 2^(E−bias) × (1 + T × 2^(−tsw)) → **normal**

##### Temel Kavramlar

| Kavram | Açıklama |
|--------|----------|
| **Domain** | Operasyonun iyi tanımlı olduğu giriş kümesi (ör: √'nin domain'i [0, +∞]) |
| **Intermediate result** | Sınırsız hassasiyetle hesaplanan ara sonuç |
| **Rounding** | Ara sonucun floating point değere dönüştürülmesi |

##### IEEE-754 İstisnalar

| İstisna | Durum | Örnek |
|---------|-------|-------|
| **Invalid operation** | Domain dışı işlem | `0 × ∞`, `sqrt(-1)` |
| **Division by zero** | Sonlu operandlarla sonsuz sonuç | `1 / 0`, `log(0)` |
| **Overflow** | Ara sonuç finite range'i aşar | Çok büyük çarpım |
| **Underflow** | Ara/sonuç subnormal | Çok küçük çarpım |
| **Inexact** | Yuvarlanan sonuç ara sonuçtan farklı | Çoğu aritmetik işlem |

#### 15.7.2 Differences from IEEE-754

WGSL, IEEE-754'ten şu şekillerde ayrılır:

| Fark | Açıklama |
|------|----------|
| **Rounding mode belirtilmez** | Ara sonuç yukarı veya aşağı yuvarlanabilir |
| **Float → Integer clamp** | Dönüştürmede değer hedef tipin aralığına sıkıştırılır (C/C++ undefined behavior yerine) |
| **Exception üretilmez** | IEEE-754 exception'ları farklı davranışlara eşlenir |
| **Signaling NaN yok** | Ara hesapta signaling NaN quiet NaN'ye dönüşebilir |
| **Sıfırın işareti yok sayılabilir** | +0 ve −0 birbirinin yerine kullanılabilir |
| **Flush to zero** | Subnormal değerler sıfıra yuvarlanabilir (belirli operasyonlarda) |
| **Accuracy tabloları** | Operasyonların doğruluğu özel tablolarla belirlenir |
| **Semantik farklılıklar** | Bazı built-in'ler IEEE-754 karşılığından farklı çalışır (ör: `fma` → `x*y+z` olarak iki yuvarlama) |

##### Finite Math Assumption

WGSL'nin en önemli kurallarından biri:

**Shader yürütmesi öncesinde** (const/override expression'lar):
- Overflow, ∞ veya NaN → **hata** üretilir (shader-creation veya pipeline-creation error)

**Shader yürütmesi sırasında** (runtime expression'lar):
- Implementasyon overflow, ∞ ve NaN'ın **olmadığını varsayabilir**
- Bu durumda sonuç **indeterminate value** olur

> ⚠️ Bu kural, `min` ve `max` gibi fonksiyonların NaN varlığında beklenen sonucu döndürmeyebileceği anlamına gelir.

##### Flush to Zero Kuralları

| Operasyon Grubu | Flush to Zero |
|-----------------|---------------|
| §15.7.4'teki accuracy tablosundaki operasyonlar | Giriş/çıkış flush edilebilir |
| Bit reinterpretation, packing, unpacking | Ara sonuçlar flush edilebilir |
| Diğer tüm operasyonlar | Subnormal değerler **korunmalıdır** |

#### 15.7.3 Floating Point Rounding and Overflow

Overflow durumunda sonuç, sonsuzluğa veya en yakın sonlu değere yuvarlanabilir. Bu, ara sonucun büyüklüğüne ve değerlendirmenin zamanlamasına bağlıdır.

**Yuvarlama kuralları (`X` ara sonuç, tip `T`):**

| Koşul | X' |
|-------|-----|
| X, T'nin sonlu aralığında | Yukarı veya aşağı yuvarlama |
| X NaN | NaN |
| MAX(T) < X < 2^(EMAX+1) | MAX(T) veya +∞ (her ikisi de olabilir) |
| 2^(EMAX+1) ≤ X | +∞ (IEEE-754 kuralı) |
| Negatif taraf | Simetrik kurallar |

**Son değer (`X''`):**
- X' sonsuz veya NaN ise:
  - const-expression → **shader-creation error**
  - override-expression → **pipeline-creation error**
  - runtime expression → **indeterminate value**
- Aksi halde: X'' = X'

#### 15.7.4 Floating Point Accuracy

##### Correctly Rounded

Tam sonuç `x` için, **correctly rounded** sonuç:
- `x` tipin bir değeriyse → `x`
- Aksi halde → `x`'in üstündeki en küçük veya altındaki en büyük representable değer

> WGSL bir **rounding mode** belirtmez — sonuç yukarı veya aşağı yuvarlanabilir.

##### ULP (Units in the Last Place)

**ULP(x)**, `x` etrafındaki iki ardışık floating point değer arasındaki **minimum mesafe**dir. Doğruluk sınırlarının ölçü birimi olarak kullanılır.

##### 15.7.4.1 Accuracy of Concrete Floating Point Expressions

**Temel aritmetik operatörler:**

| İfade | f32 Doğruluğu | f16 Doğruluğu |
|-------|---------------|---------------|
| `x + y` | Correctly rounded | Correctly rounded |
| `x - y` | Correctly rounded | Correctly rounded |
| `x * y` | Correctly rounded | Correctly rounded |
| `x / y` | 2.5 ULP (\|y\| ∈ [2⁻¹²⁶, 2¹²⁶]) | 2.5 ULP (\|y\| ∈ [2⁻¹⁴, 2¹⁴]) |
| `x % y` | Inherited: `x - y * trunc(x/y)` | Inherited: `x - y * trunc(x/y)` |
| `-x` | Correctly rounded | Correctly rounded |
| Karşılaştırmalar | Correct result | Correct result |

**Seçili built-in fonksiyonların doğruluğu:**

| Fonksiyon | f32 | f16 |
|-----------|-----|-----|
| `abs(x)` | Correctly rounded | Correctly rounded |
| `acos(x)` | max(abs err 6.77×10⁻⁵, `atan2(sqrt(1-x*x),x)`) | max(abs err 3.91×10⁻³, inherited) |
| `asin(x)` | max(abs err 6.81×10⁻⁵, inherited) | max(abs err 3.91×10⁻³, inherited) |
| `atan(x)` | 4096 ULP | 5 ULP |
| `atan2(y,x)` | 4096 ULP (\|x\| ∈ [2⁻¹²⁶, 2¹²⁶]) | 5 ULP (\|x\| ∈ [2⁻¹⁴, 2¹⁴]) |
| `cos(x)` | abs err ≤ 2⁻¹¹ (x ∈ [−π, π]) | abs err ≤ 2⁻⁷ |
| `sin(x)` | abs err ≤ 2⁻¹¹ (x ∈ [−π, π]) | abs err ≤ 2⁻⁷ |
| `exp(x)` | (3 + 2\|x\|) ULP | (1 + 2\|x\|) ULP |
| `exp2(x)` | (3 + 2\|x\|) ULP | (1 + 2\|x\|) ULP |
| `log(x)` | abs err ≤ 2⁻²¹ ([0.5,2.0]); 3 ULP (dışı) | abs err ≤ 2⁻⁷; 3 ULP |
| `log2(x)` | abs err ≤ 2⁻²¹ ([0.5,2.0]); 3 ULP (dışı) | abs err ≤ 2⁻⁷; 3 ULP |
| `sqrt(x)` | Inherited: `1.0/inverseSqrt(x)` | Inherited |
| `inverseSqrt(x)` | 2 ULP | 2 ULP |
| `pow(x,y)` | Inherited: `exp2(y * log2(x))` | Inherited |
| `fma(x,y,z)` | Inherited: `x * y + z` | Inherited |
| `determinant(m)` | Infinite ULP | Infinite ULP |
| `dpdx/dpdy/fwidth` | Infinite ULP | Infinite ULP |

**Inherited accuracy** — bir operasyonun doğruluğu, alternatif bir WGSL ifadesinin doğruluğundan devralınır. Örneğin `tan(x)` → `sin(x) / cos(x)`.

> **Not:** `determinant` ve derivative fonksiyonları için sonlu bir hata sınırı **yoktur**. Bu, alttaki GPU implementasyonlarının aynı eksikliğini yansıtır.

##### 15.7.4.2 Accuracy of AbstractFloat Expressions

AbstractFloat operasyonlarının doğruluğu:

| Kural | Koşul |
|-------|-------|
| Correct result gerekli | f32 karşılığı correct result gerektiriyorsa |
| Correctly rounded gerekli | f32 karşılığı correctly rounded gerektiriyorsa |
| `fract(x)` | `x - floor(x)` olarak inherited (AbstractFloat aritmetiği ile) |
| Diğer (ör: abs/rel/inherited err) | **Unbounded** (sınırsız) — ancak f32'deki hata sınırını aşmaması **önerilir** |

> **ULP çevirisi:** 1 ULP (f32) = 2²⁹ ULP (AbstractFloat/binary64) — significand'daki bit farkından dolayı.

#### 15.7.5 Reassociation and Fusion

##### Reassociation (Yeniden İlişkilendirme)

Operasyonların matematiksel olarak eşdeğer, ancak farklı sırada hesaplanması:

```wgsl
(a + b) + c  →  a + (b + c)    // Reassociation
(a - b) + c  →  (a + c) - b    // Reassociation
(a * b) / c  →  (a / c) * b    // Reassociation
```

> ⚠️ Floating point aritmetiğinde sonuç farklı olabilir (hassasiyet kaybı veya overflow). Yine de **implementasyon reassociation yapabilir**.

##### Fusion (Birleştirme)

Birden fazla operasyonun tek bir daha doğru operasyona birleştirilmesi:
- Birleştirilen ifade, orijinalinden **en az eşit doğruluğa** sahip olmalıdır
- Örnek: FMA (fused multiply-add) ayrı çarpma ve toplamadan daha doğru olabilir

#### 15.7.6 Floating Point Conversion

##### Float → Integer

```
X (float) → T (integer tipi)
```

| Durum | Sonuç |
|-------|-------|
| X NaN | Indeterminate value (T tipinde) |
| X, T'de tam temsil edilebilir | O değer |
| Diğer | `truncate(X)` sonucuna en yakın, hem T'de hem de kaynak float tipinde representable değer |

> **Pratikte:** Değer, hedef tipin aralığına **clamp** edilir, sonra sıfıra doğru yuvarlanır. Bu, C/C++'taki undefined behavior yerine WGSL'nin **tanımlı davranış** garantisidir.

**Örnekler:**
- `3.9f → u32` = `3u`
- `-1f → u32` = `0u`
- `1e20f → u32` = `4294967040u` (en büyük f32-representable u32 değeri)
- `-3.9f → i32` = `-3i`

##### Integer/Float → Float

| Durum | Sonuç |
|-------|-------|
| X NaN (kaynak float ise) | NaN (hedef tipte) |
| X, T'de tam temsil edilebilir | O değer |
| X, T'deki iki komşu değer arasında | İkisinden biri (belirsiz – her instance farklı seçebilir) |
| X, T'nin sonlu aralığı dışında | const-expr → shader-creation error; override-expr → pipeline-creation error; runtime → clamp veya ∞ |

> **Not:** `i32`/`u32` → `f32` dönüşümünde değer her zaman f32'nin aralığı içindedir. Ancak büyük tamsayılar hassasiyet kaybına uğrayabilir (ör: ardışık tamsayılar 2²⁵'ten itibaren aynı f32 değerine çarpışır).

#### 15.7.7 Domains of Floating Point Expressions and Built-in Functions

Her floating point operasyonun bir **domain**'i (tanım kümesi) vardır:

- Kısıtlama belirtilmemişse: domain **total** (tüm sonlu ve sonsuz girdiler)
- Aksi halde: açık domain listelenir (ör: `acos` domain'i [−1, 1])

**Component-wise** operasyonlar için yalnızca skaler durum tanımlanır; vektör durumu component-wise semantiklerden türetilir.

**Inherited accuracy** olan operasyonlarda domain:
- **Açık** olarak tanımlanır, veya
- **Implied from linear terms** — "inherited from" ifadesindeki toplama, çıkarma ve çarpma operasyonlarının domain kısıtlamalarının birleştirilmesiyle türetilir.

**Örnek:** `dot(a, b)` (2-element) = `a[0]*b[0] + a[1]*b[1]`:
- Çarpma: `0 × ∞` ve `∞ × 0` hariç
- Toplama: Zıt işaretli sonsuzların toplamı hariç
- Sonuç domain: Bu dışlamaların birleşimi

---

> **Önceki:** [← GPU Arayüzü ve Bellek](05-gpu-arayuzu-ve-bellek.md) · **Sonraki:** [Built-in Kütüphanesi →](07-built-in-kutuphanesi.md)
