---
title: GPU Arayüzü ve Bellek
section: 12-14
source: "W3C WGSL Spec §12–§14"
---

# 5. GPU Arayüzü ve Bellek

> Shader'ın dış dünyayla bağlantısı: 15 attribute, entry point'ler, shader interface, bellek düzeni ve bellek modeli.

---

## §12 Attributes

**Attribute**, bir WGSL nesnesi üzerinde ek bilgi veya kısıtlama belirten bir meta-veri etiketidir. `@` sembolü ile başlar ve isteğe bağlı olarak parantez içinde parametre alabilir.

```wgsl
@attribute
@attribute(param)
@attribute(param1, param2)
```

### 12.1 `align`

Bir yapı üyesinin bellekteki **hizalama** (alignment) gereksinimini belirtir. Yalnızca yapı üye bildirimlerine uygulanabilir.

- **Parametre:** Pozitif, 2'nin kuvveti olan bir tamsayı (ör: 4, 8, 16, 32...)
- Belirtilen değer, üyenin tipinin doğal hizalaması (`AlignOf`) kadar veya daha büyük olmalıdır.

```wgsl
struct MyStruct {
  @align(16) position: vec3<f32>,  // 16-byte sınırına hizala
  @align(4)  value: f32            // 4-byte sınırına hizala
}
```

### 12.2 `binding`

Bir kaynak değişkeninin **bağlama numarasını** (binding number) belirtir. `@group` ile birlikte kullanılarak kaynağın pipeline üzerindeki adresini tanımlar.

- **Parametre:** Negatif olmayan bir tamsayı (`i32` veya `u32` olarak temsil edilebilir)
- Yalnızca `uniform`, `storage` veya `handle` adres uzayındaki modül kapsamı değişkenlere uygulanır.

```wgsl
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;
@group(0) @binding(2) var mySampler: sampler;
```

### 12.3 `blend_src`

Fragment shader çıktısında **çift kaynaklı karıştırma** (dual source blending) için kaynak indeksini belirtir.

- **Parametre:** `0` veya `1`
- Yalnızca fragment shader çıktı yapı üyelerine uygulanır.
- Kullanıldığında, `@location(0) @blend_src(0)` ve `@location(0) @blend_src(1)` olmak üzere tam olarak iki giriş gerekir ve her ikisi aynı veri tipinde olmalıdır.

### 12.4 `builtin`

İlgili nesnenin bir **yerleşik değer** (built-in value) olduğunu belirtir. Shader aşamasına özgü sistem tarafından sağlanan veya üretilen değerlere erişim sağlar.

- **Parametre:** Yerleşik değer adı (ör: `position`, `vertex_index`, `global_invocation_id`)
- Giriş noktası parametrelerine, dönüş tiplerine veya yapı üyelerine uygulanabilir.

```wgsl
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
  return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
```

### 12.5 `const`

Bir fonksiyon parametresinin **derleme zamanı sabiti** (const-expression) olarak değerlendirilmesi gerektiğini belirtir. Bu attribute ile işaretlenmiş parametreye yalnızca const-expression geçirilebilir.

```wgsl
fn clampedValue(@const min_val: i32, x: i32) -> i32 {
  // min_val derleme zamanında bilinen bir sabit
  return max(min_val, x);
}
```

### 12.6 `diagnostic`

**Tanılama kurallarını** (diagnostic rules) yönetir. Belirli uyarı veya hata mesajlarının önem derecesini kontrol etmek için kullanılır. Fonksiyon bildirimlerine, compound statement'lara veya modül kapsamında `diagnostic` direktifi olarak uygulanabilir.

```wgsl
@diagnostic(off, derivative_uniformity)
fn myFragment() { /* ... */ }
```

### 12.7 `group`

Bir kaynak değişkeninin **bağlama grubu** numarasını belirtir. `@binding` ile birlikte kullanılır.

- **Parametre:** Negatif olmayan bir tamsayı
- Aynı shader'daki iki farklı kaynak değişkeni aynı `(group, binding)` çiftine sahip olamaz.

### 12.8 `id`

Bir `override` bildiriminin pipeline-constant ID'sini belirtir.

- **Parametre:** Negatif olmayan bir tamsayı (`u16` aralığında, 0–65535)
- Aynı shader'daki iki farklı override bildirimi aynı `@id` değerine sahip olamaz.

```wgsl
@id(0) override blockSize: u32 = 64;
@id(1) override threshold: f32 = 0.5;
```

### 12.9 `interpolate`

Kullanıcı tanımlı G/Ç (input/output) verilerinin **interpolasyon** yöntemini kontrol eder.

- **Parametreler:** İnterpolasyon tipi ve isteğe bağlı örnekleme modu
- Yalnızca skaler veya vektör tipindeki `@location` ile etiketlenmiş G/Ç'ye uygulanır.

**İnterpolasyon tipleri:**

| Tip | Açıklama |
|-----|----------|
| `perspective` | Perspektif doğru interpolasyon (varsayılan) |
| `linear` | Lineer, perspektif olmayan interpolasyon |
| `flat` | İnterpolasyon yok |

**Örnekleme modları:**

| Mod | Açıklama |
|-----|----------|
| `center` | Piksel merkezinde (varsayılan, `perspective`/`linear` için) |
| `centroid` | Primitif tarafından kaplanan örneklemlerin kesişiminde |
| `sample` | Her örnek başına (fragment shader'ı her örnek için çalıştırır) |
| `first` | Primitifin ilk vertex'inin değeri (varsayılan, `flat` için) |
| `either` | İlk veya son vertex (uygulama bağımlı) |

```wgsl
struct VertexOutput {
  @builtin(position) pos: vec4<f32>,
  @location(0) @interpolate(perspective, center) color: vec3<f32>,
  @location(1) @interpolate(flat) id: u32
}
```

> **Kural:** Tamsayı tipindeki vertex çıktıları ve fragment girdileri her zaman `@interpolate(flat)` olarak belirtilmelidir.

### 12.10 `invariant`

Bir `@builtin(position)` değerinin, aynı girdiler ve aynı pipeline durumu için her çağrıda **aynı** sonucu üreteceğini garanti eder. Derinlik testi tutarsızlıklarını önlemek için kullanılır.

```wgsl
struct VertexOut {
  @builtin(position) @invariant pos: vec4<f32>
}
```

### 12.11 `location`

Bir giriş noktasının **kullanıcı tanımlı G/Ç konumunu** belirtir. Her locationa bir skaler veya vektör tipi atanır.

- **Parametre:** Negatif olmayan bir tamsayı
- Girdiler ve çıktılar için location numaralaması **bağımsızdır** (aynı numara hem girdi hem çıktıda kullanılabilir).
- Aynı yöndeki (girdi veya çıktı) iki öğe aynı location değerini paylaşamaz.

```wgsl
struct FragInput {
  @location(0) color: vec4<f32>,
  @location(1) uv: vec2<f32>
}

@fragment
fn fs_main(input: FragInput) -> @location(0) vec4<f32> {
  return input.color;
}
```

### 12.12 `must_use`

Bir fonksiyonun dönüş değerinin **kullanılması gerektiğini** belirtir. Kullanılmayan bir dönüş değeri shader-creation hatasına neden olur.

```wgsl
@must_use
fn computeValue() -> f32 {
  return 42.0;
}

fn caller() {
  let v = computeValue();  // ✓ Geçerli
  // computeValue();       // ✗ Hata: dönüş değeri kullanılmadı
}
```

### 12.13 `size`

Bir yapı üyesinin bellekte kapladığı **byte boyutunu** belirtir. Varsayılan boyuttan daha büyük bir değer ayarlanabilir (fazlalık dolgu olur).

- **Parametre:** Pozitif bir tamsayı (üyenin tipinin doğal boyutu kadar veya daha büyük olmalı)

```wgsl
struct Light {
  @size(16) intensity: f32,   // 4 byte veri + 12 byte dolgu = 16 byte
  color: vec3<f32>
}
```

### 12.14 `workgroup_size`

Compute shader'ın **workgroup boyutlarını** belirtir. Yalnızca `@compute` giriş noktalarına uygulanır.

- **Parametreler:** 1, 2 veya 3 pozitif tamsayı (x, y, z boyutları)
- Belirtilmeyen boyutlar varsayılan olarak `1` kabul edilir.
- Değerler `const-expression` veya `override-expression` olabilir.

```wgsl
@compute @workgroup_size(256)
fn main1() { /* 256×1×1 workgroup */ }

@compute @workgroup_size(16, 16)
fn main2() { /* 16×16×1 workgroup */ }

@compute @workgroup_size(8, 8, 4)
fn main3() { /* 8×8×4 = 256 iş parçacığı */ }

// Override ile dinamik boyut
override blockSize: u32 = 64;
@compute @workgroup_size(blockSize)
fn main4() { /* Pipeline oluşturma zamanında belirlenir */ }
```

### 12.15 Shader Stage Attributes

Bir fonksiyonu belirli bir **shader aşaması** için **giriş noktası** (entry point) olarak işaretler. Her fonksiyonda en fazla bir shader stage attribute bulunabilir.

#### 12.15.1 `@vertex`

Fonksiyonu **vertex shader** giriş noktası olarak işaretler. Vertex verilerini işler ve kırpma uzayı koordinatlarını üretir.

#### 12.15.2 `@fragment`

Fonksiyonu **fragment shader** giriş noktası olarak işaretler. Her fragment (piksel adayı) için renk ve derinlik değerleri üretir.

#### 12.15.3 `@compute`

Fonksiyonu **compute shader** giriş noktası olarak işaretler. Genel amaçlı GPU hesaplamalar için kullanılır; `@workgroup_size` ile birlikte kullanılmalıdır.

```wgsl
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
  return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
  return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // Genel amaçlı hesaplama
}
```

---

## §13 Entry Points

### 13.1 Shader Stages

WGSL, GPU pipeline'ının üç aşamasını destekler:

| Aşama | Attribute | Açıklama |
|-------|-----------|----------|
| **Vertex** | `@vertex` | Her vertex için çalışır; konum, normal vb. dönüştürür |
| **Fragment** | `@fragment` | Her fragment (piksel adayı) için çalışır; renk/derinlik üretir |
| **Compute** | `@compute` | Render pipeline'dan bağımsız genel amaçlı hesaplama |

Vertex ve fragment aşamaları bir **render pipeline** oluşturur. Compute aşaması ise bağımsız bir **compute pipeline** kullanır.

### 13.2 Entry Point Declaration

Bir giriş noktası, shader stage attribute'u ile işaretlenmiş bir kullanıcı tanımlı fonksiyondur.

- **Parametreler:** Shader aşaması girdilerini tanımlar
- **Dönüş tipi:** Shader aşaması çıktılarını tanımlar
- Giriş noktası fonksiyonunun çağırdığı tüm fonksiyonlar **shader'ın fonksiyonları** olarak kabul edilir.

```wgsl
@vertex
fn vs_main(
  @builtin(vertex_index) my_index: u32,
  @location(0) my_position: vec4<f32>
) -> @builtin(position) vec4<f32> {
  return my_position;
}
```

**Kısıtlamalar:**
- Giriş noktaları diğer giriş noktalarını **çağıramaz**.
- Aynı modülde birden fazla giriş noktası bildirilebilir.
- `@compute` giriş noktaları `@workgroup_size` attribute'u gerektirir.

### 13.3 Shader Interface

Bir shader'ın **arayüzü** (interface), dış dünya ile veri alışverişi yapmasını sağlayan bileşenlerden oluşur:

1. **Shader Stage Inputs/Outputs** — Aşamalar arası veri akışı
2. **Override Declarations** — Pipeline oluşturma zamanında belirlenen sabitler
3. **Resources** — Buffer, texture ve sampler gibi dış kaynaklar

#### 13.3.1 Inter-stage Input and Output Interface

Shader aşamaları arasındaki G/Ç arayüzü, `@builtin` ve `@location` attribute'ları ile tanımlanır. Vertex shader çıktıları, rasterizasyondan sonra fragment shader girdilerine eşlenir.

##### 13.3.1.1 Built-in Inputs and Outputs

Yerleşik değerler, GPU pipeline tarafından otomatik olarak sağlanan veya tüketilen sistem değerleridir. `@builtin(value_name)` ile erişilir.

| Yerleşik Değer | Aşama | Yön | Tip | Açıklama |
|----------------|-------|-----|-----|----------|
| `vertex_index` | Vertex | Girdi | `u32` | Mevcut vertex indeksi |
| `instance_index` | Vertex | Girdi | `u32` | Mevcut instance indeksi |
| `clip_distances` | Vertex | Çıktı | `array<f32, N>` | Kullanıcı tanımlı kırpma mesafeleri |
| `position` | Vertex/Fragment | Çıktı/Girdi | `vec4<f32>` | Kırpma uzayı konumu (V) / Framebuffer konumu (F) |
| `front_facing` | Fragment | Girdi | `bool` | Primitifin ön yüze bakıp bakmadığı |
| `frag_depth` | Fragment | Çıktı | `f32` | Fragment derinlik değerini geçersiz kılar |
| `sample_index` | Fragment | Girdi | `u32` | Mevcut çoklu örnekleme indeksi |
| `sample_mask` | Fragment | Girdi/Çıktı | `u32` | Örnek kapsama maskesi |
| `primitive_index` | Fragment | Girdi | `u32` | Mevcut primitif indeksi |
| `local_invocation_id` | Compute | Girdi | `vec3<u32>` | Workgroup içindeki 3D iş parçacığı ID'si |
| `local_invocation_index` | Compute | Girdi | `u32` | Workgroup içindeki düzleştirilmiş indeks |
| `global_invocation_id` | Compute | Girdi | `vec3<u32>` | Tüm dispatch içindeki 3D ID |
| `workgroup_id` | Compute | Girdi | `vec3<u32>` | Mevcut workgroup'un 3D indeksi |
| `num_workgroups` | Compute | Girdi | `vec3<u32>` | Dispatch edilen toplam workgroup sayısı |

**Subgroup yerleşik değerleri** (opsiyonel özellik):

| Yerleşik Değer | Aşama | Yön | Tip | Açıklama |
|----------------|-------|-----|-----|----------|
| `subgroup_invocation_id` | Compute/Fragment | Girdi | `u32` | Subgroup içindeki iş parçacığı indeksi |
| `subgroup_size` | Compute/Fragment | Girdi | `u32` | Subgroup boyutu |
| `subgroup_id` | Compute | Girdi | `u32` | Workgroup içindeki subgroup indeksi |
| `num_subgroups` | Compute | Girdi | `u32` | Workgroup'taki toplam subgroup sayısı |

##### 13.3.1.2 User-defined Inputs and Outputs

Kullanıcı tanımlı G/Ç, `@location` attribute'u ile numaralandırılmış veri yuvaları üzerinden gerçekleşir. Kabul edilen tipler: skaler, vektör (en fazla 4 bileşenli) veya bu tipleri içeren yapılar.

```wgsl
struct VertexOutput {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec3<f32>,
  @location(1) @interpolate(flat) id: u32
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VertexOutput {
  var out: VertexOutput;
  out.pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);
  out.color = vec3<f32>(1.0, 0.0, 0.0);
  out.id = vi;
  return out;
}
```

> **Not:** Kullanıcı tanımlı G/Ç yapıları iç içe yerleştirilemez (nested struct yasak).

##### 13.3.1.3 Input-output Locations

- Her `@location` değerine bir skaler veya vektör tipi atanır.
- Aynı yöndeki (girdi veya çıktı) iki öğe aynı `@location` değerini paylaşamaz.
- Girdi ve çıktı location numaraları **birbirinden bağımsızdır.**
- Kullanılabilir location sayısı WebGPU API tarafından belirlenir.

```wgsl
// in1: location 0 ve 1; in2: location 2; çıktı: location 0
@fragment
fn fragShader(in1: A, @location(2) in2: f32) -> @location(0) vec4<f32> {
  // Girdi location 0 ile çıktı location 0 çakışmaz
  // ...
}
```

##### 13.3.1.4 Interpolation

İnterpolasyon, vertex shader çıktılarının fragment shader girdilerine nasıl dönüştürüleceğini kontrol eder.

**Varsayılanlar:**
- Kayan noktalı tipler: `@interpolate(perspective, center)`
- `flat` tipi: `@interpolate(flat, first)`

**Kombinasyon kuralları:**

| İnterpolasyon Tipi | İzin Verilen Örnekleme Modları |
|---------------------|-------------------------------|
| `perspective` | `center`, `centroid`, `sample` |
| `linear` | `center`, `centroid`, `sample` |
| `flat` | `first`, `either` |

> **Önemli:** Render pipeline'da, her kullanıcı tanımlı fragment girdisinin interpolasyon özellikleri, aynı location'daki vertex çıktısıyla **eşleşmelidir**. Aksi halde pipeline-creation hatası oluşur.

#### 13.3.2 Resource Interface

**Kaynak** (resource), shader aşaması dışındaki verilere erişim sağlayan bir nesnedir. Tüm invocation'lar tarafından paylaşılır.

**Dört tür kaynak vardır:**

1. **Uniform buffer** — Sabit, salt-okunur veri blokları
2. **Storage buffer** — Okuma/yazma yeteniğine sahip büyük veri blokları
3. **Texture** — Görüntü verileri
4. **Sampler** — Doku örnekleme parametreleri

Her kaynak değişkeni `@group` ve `@binding` attribute'ları ile bildirilmelidir:

```wgsl
@group(0) @binding(0) var<uniform> camera: CameraData;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(1) @binding(0) var myTexture: texture_2d<f32>;
@group(1) @binding(1) var mySampler: sampler;
```

**WebGPU bağlama tipi uyumluluğu:**

| WGSL Kaynağı | WebGPU Binding Tipi |
|--------------|---------------------|
| Uniform buffer | `"uniform"` |
| Storage buffer (read_write) | `"storage"` |
| Storage buffer (read) | `"read-only-storage"` |
| Sampler | `"filtering"` / `"non-filtering"` |
| Sampler (comparison) | `"comparison"` |
| Sampled/depth/multisampled texture | `GPUTextureSampleType` |
| Write-only storage texture | `"write-only"` |
| Read-write storage texture | `"read-write"` |
| Read-only storage texture | `"read-only"` |
| External texture | `externalTexture` |

#### 13.3.3 Resource Layout Compatibility

Shader'ın kaynak arayüzü, pipeline'ın düzeni ile **uyumlu** olmalıdır. Uyumsuzluk durumunda **pipeline-creation hatası** oluşur.

#### 13.3.4 Buffer Binding Determines Runtime-Sized Array Element Count

Bir storage buffer değişkeni **runtime-sized array** içerdiğinde, dizideki eleman sayısı bağlı buffer'ın boyutundan hesaplanır:

```
NRuntime = truncate((EBBS − array_offset) ÷ array_stride)
```

- **EBBS:** Effective buffer binding size (bağlı buffer'ın etkin boyutu)
- **array_offset:** Değişken içinde dizinin başlangıç byte ofseti
- **array_stride:** Dizi elemanlarının adım boyutu (`StrideOf`)

```wgsl
@group(0) @binding(1) var<storage> weights: array<f32>;
// EBBS = 1024 → NRuntime = truncate(1024 / 4) = 256
// EBBS = 1025 → NRuntime = truncate(1025 / 4) = 256
// EBBS = 1028 → NRuntime = truncate(1028 / 4) = 257
```

**Yapı içindeki runtime-sized array:**

```wgsl
struct PointLight {                          //             align(16) size(32)
  position: vec3f,                           // offset(0)   align(16) size(12)
  // -- örtük hizalama dolgusu --            // offset(12)            size(4)
  color: vec3f,                              // offset(16)  align(16) size(12)
  // -- örtük yapı boyutu dolgusu --         // offset(28)            size(4)
}

struct LightStorage {                        //             align(16)
  pointCount: u32,                           // offset(0)   align(4)  size(4)
  // -- örtük hizalama dolgusu --            // offset(4)             size(12)
  point: array<PointLight>,                  // offset(16)  align(16) stride(32)
}

@group(0) @binding(1) var<storage> lights: LightStorage;
// EBBS = 1024 → NRuntime = truncate((1024 - 16) / 32) = 31
// EBBS = 1040 → NRuntime = truncate((1040 - 16) / 32) = 32
```

Shader, `arrayLength` yerleşik fonksiyonu ile `NRuntime` hesaplayabilir.

---

## §14 Memory

### 14.1 Memory Locations

Bellek, 8-bitlik **bellek konumlarından** (memory locations) oluşan soyut bir dizidir. Her değişken, bellek konumlarından oluşan ayrı bir kümeye eşlenir. Bellek operasyonları bu konumlarla etkileşime girer; **dolgu** (padding) bellek konumlarına erişim yapılmaz.

### 14.2 Memory Access Mode

**Erişim modu**, bir bellek referansı üzerinde hangi operasyonların gerçekleştirilebileceğini belirler:

| Erişim Modu | Açıklama |
|-------------|----------|
| `read` | Yalnızca okuma. Bellek konumları okunabilir ancak yazılamaz. |
| `write` | Yalnızca yazma. Bellek konumları yazılabilir ancak okunamaz. |
| `read_write` | Okuma ve yazma. Hem okunabilir hem yazılabilir. |

- **Okuma erişimi** (read access): Bir bellek görünümünden (memory view) değer okumak
- **Yazma erişimi** (write access): Bir bellek görünümüne değer yazmak
- Erişim modu, referans tipi `ref<AS, T, AM>` ve pointer tipi `ptr<AS, T, AM>` üzerinde `AM` parametresi ile belirtilir.

### 14.3 Address Spaces

Bellek konumları **adres uzaylarına** (address spaces) ayrılır. Her adres uzayı farklı yaşam süresi, erişim kuralları ve kullanım alanına sahiptir.

| Adres Uzayı | Varsayılan Erişim | Yaşam Süresi | Açıklama | Değişken Bildirimi |
|-------------|-------------------|-------------|----------|-------------------|
| `function` | `read_write` | Fonksiyon çağrısı | Fonksiyon-yerel değişkenler | `var x: T` (fonksiyon gövdesinde) |
| `private` | `read_write` | Invocation | İş parçacığı başına özel veri | `var<private> x: T` |
| `workgroup` | `read_write` | Workgroup | Workgroup içi paylaşılan bellek | `var<workgroup> x: T` |
| `uniform` | `read` | Pipeline | Sabit buffer (tüm invocation'lar okur) | `var<uniform> x: T` |
| `storage` | `read` | Pipeline | Büyük veri buffer'ları (R veya R/W) | `var<storage> x: T` veya `var<storage, read_write> x: T` |
| `handle` | `read` | Pipeline | Texture ve sampler kaynakları | `var x: texture_2d<f32>` |

**Önemli kısıtlamalar:**
- `function` adres uzayı yalnızca fonksiyon kapsamında bildirilir; adres uzayı açıkça belirtilmez.
- `workgroup` değişkenleri **yalnızca compute shader'larda** kullanılabilir.
- `uniform` adres uzayı **runtime-sized array** içeremez.
- `handle` adres uzayı doğrudan belirtilemez; texture ve sampler tipleri otomatik olarak bu uzaya yerleşir.
- `storage` adres uzayı yalnızca **host-shareable** tipler için kullanılabilir.

### 14.4 Memory Layout

Bellek düzeni, değerlerin bellekte nasıl yerleştirildiğini belirler. **Host-shareable** tipler (CPU-GPU arası paylaşılabilen tipler) için düzen kuralları kesindir.

#### 14.4.1 Alignment and Size

Her tipin bir **hizalama** (`AlignOf`) ve **boyut** (`SizeOf`) değeri vardır:

| Tip | AlignOf (byte) | SizeOf (byte) |
|-----|----------------|---------------|
| `bool` | 4 | 4 |
| `i32`, `u32`, `f32` | 4 | 4 |
| `f16` | 2 | 2 |
| `vec2<T>` | 2 × AlignOf(T) | 2 × SizeOf(T) |
| `vec3<T>` | 4 × AlignOf(T) | 3 × SizeOf(T) |
| `vec4<T>` | 4 × AlignOf(T) | 4 × SizeOf(T) |
| `matCxR<T>` | AlignOf(vecR<T>) | C × StrideOf(vecR<T>) |
| `array<T, N>` | AlignOf(T) | N × StrideOf(array<T,N>) |

**Yardımcı fonksiyonlar:**
- `StrideOf(array<T, N>)` = `roundUp(AlignOf(T), SizeOf(T))` — Dizi elemanlarının adım boyutu
- `roundUp(k, n)` = `⌈n/k⌉ × k` — `n`'yi `k`'nin katına yuvarlama
- `OffsetOfMember(S, i)` — Yapı `S`'nin `i`. üyesinin byte ofseti

#### 14.4.2 Structure Member Layout

Yapı üyeleri bellekte sıralı olarak yerleştirilir. Her üyenin ofseti, hizalama gereksinimine göre hesaplanır:

```
OffsetOfMember(S, 0) = 0  (yapının içindeyse @align ile geçersiz kılınabilir)
OffsetOfMember(S, i) = roundUp(AlignOf(Mi), OffsetOfMember(S, i−1) + SizeOf(Mi−1))
```

**Yapının toplam boyutu:** Son üyenin ofset+boyutundan sonra, yapının hizalama değerine yuvarlanmış değer.

```wgsl
struct A {
  u: f32,           // offset(0)   align(4)  size(4)
  v: f32,           // offset(4)   align(4)  size(4)
  w: vec2<f32>,     // offset(8)   align(8)  size(8)
  x: f32            // offset(16)  align(4)  size(4)
}                   // SizeOf(A) = roundUp(8, 20) = 24 (4 byte dolgu)
```

`@align` ve `@size` attribute'ları ile üye düzeni özelleştirilebilir:

```wgsl
struct B {
  @align(16) a: f32,   // offset(0)   boyut(4)
  @size(32)  b: f32,   // offset(16)  boyut(32) — 28 byte dolgu
  c: f32               // offset(48)
}
```

#### 14.4.3 Array Layout Examples

Dizi elemanları `StrideOf` adım boyutu ile sıralı yerleştirilir:

```
element_i_offset = base_offset + i × StrideOf(array<T, N>)
```

Örnek: `array<vec3<f32>, 4>` dizisi:

- `AlignOf(vec3<f32>)` = 16
- `SizeOf(vec3<f32>)` = 12
- `StrideOf` = `roundUp(16, 12)` = 16
- Her eleman 16 byte yer kaplar (12 byte veri + 4 byte dolgu)

#### 14.4.4 Internal Layout of Values

Host-shareable değerlerin bellekte little-endian formatında saklandığı belirtilir. Anahtar düzen kuralları:

- **`u32` / `i32`:** 4 byte, little-endian. `i32` ikiye tümleyici kullanır.
- **`f32`:** IEEE-754 binary32 (1 işaret + 8 üssü + 23 kesir biti), little-endian.
- **`f16`:** IEEE-754 binary16 (1 işaret + 5 üssü + 10 kesir biti), little-endian.
- **Vektör:** Bileşenler sıralı yerleşir: `V.x` offset `k`, `V.y` offset `k + SizeOf(T)`, vb.
- **Matris:** Sütun vektörü `i`, offset `k + i × AlignOf(vecR<T>)` konumuna yerleşir.
- **Dizi:** Eleman `i`, offset `k + i × StrideOf(A)` konumuna yerleşir.
- **Yapı:** Üye `i`, offset `k + OffsetOfMember(S, i)` konumuna yerleşir.
- **Atomic:** Alt tipteki (`T`) değer ile aynı düzene sahiptir.

#### 14.4.5 Address Space Layout Constraints

`storage` ve `uniform` adres uzaylarının farklı düzen kısıtlamaları vardır.

**`RequiredAlignOf(S, C)`** — `S` tipinin `C` adres uzayındaki zorunlu hizalama değeri:

| Tip | Storage / Diğer | Uniform (`uniform_buffer_standard_layout` yok) |
|-----|-----------------|------------------------------------------------|
| Skaler, vektör, matris | `AlignOf(S)` | `AlignOf(S)` |
| `array<T, N>` | `AlignOf(S)` | `roundUp(16, AlignOf(S))` |
| `array<T>` (runtime) | `AlignOf(S)` | Yasak |
| `struct S` | `AlignOf(S)` | `roundUp(16, AlignOf(S))` |

**Uniform adres uzayı ek kısıtlamaları** (`uniform_buffer_standard_layout` desteklenmediğinde):

1. Dizi eleman adımları (`StrideOf`) 16'nın katı olmalıdır.
2. Yapı tipi üye ile sonraki üye arasındaki fark en az `roundUp(16, SizeOf(S))` olmalıdır.

```wgsl
// ✗ Geçersiz: stride 4, 16'nın katı değil
struct invalid_stride {
  a: array<f32, 8>  // stride = 4
}
@group(0) @binding(0) var<uniform> bad: invalid_stride;

// ✓ Geçerli: wrapper ile stride 16 yapılır
struct wrapped_f32 {
  @size(16) elem: f32
}
struct valid_stride {
  a: array<wrapped_f32, 8>  // stride = 16
}
@group(0) @binding(1) var<uniform> good: valid_stride;
```

### 14.5 Memory Model

WGSL, genel olarak **Vulkan Memory Model**'ini takip eder. Bu bölüm, WGSL programlarının Vulkan Memory Model'e nasıl eşlendiğini açıklar.

#### 14.5.1 Memory Operation

- **Okuma erişimi** (read access) = Vulkan Memory Model'deki "memory read operation"
- **Yazma erişimi** (write access) = Vulkan Memory Model'deki "memory write operation"

**Okuma erişimi oluşturan durumlar:**
- Load Rule değerlendirmesi (bellek referansından değer okuma)
- Texture yerleşik fonksiyonları (`textureStore`, `textureDimensions`, `textureNumLayers`, `textureNumLevels`, `textureNumSamples` hariç)
- `atomicLoad` dahil atomik fonksiyonlar (`atomicStore` hariç)
- `workgroupUniformLoad` fonksiyonu
- Bileşik atama ifadesinin sol tarafı

**Yazma erişimi oluşturan durumlar:**
- Atama ifadesi (basit veya bileşik)
- `textureStore` fonksiyonu
- `atomicStore` dahil atomik fonksiyonlar (`atomicLoad` hariç)

> **Not:** `atomicCompareExchangeWeak` yalnızca `exchanged` üyesi `true` ise yazma gerçekleştirir. Read-modify-write atomik fonksiyonlar hem okuma hem yazma erişimi içerir.

Bir bellek operasyonu, ilgili **memory view** ile ilişkili bellek konumlarına erişir. Örneğin, bir yapının belirli bir üyesini okumak yalnızca o üyenin bellek konumlarını etkiler.

> **Not:** Bir vektör bileşenine yazma, vektörün **tüm** bellek konumlarına erişebilir.

#### 14.5.2 Memory Model Reference

- Her modül kapsamı **kaynak değişkeni**, benzersiz `(group, binding)` çifti için bir memory model reference oluşturur.
- `function`, `private` ve `workgroup` adres uzaylarındaki değişkenler, yaşam süreleri boyunca benzersiz bir memory model reference oluşturur.

#### 14.5.3 Scoped Operations

Kapsamlı operasyonlar iki küme üzerinde etkili olur:

- **Memory scope:** Bellek güncellemelerinin görünür olacağı invocation kümesi
- **Execution scope:** Operasyona katılabilecek invocation kümesi

**Atomik fonksiyonların bellek kapsamı:**
- Pointer `workgroup` adres uzayındaysa → `Workgroup` kapsamı
- Pointer `storage` adres uzayındaysa → `QueueFamily` kapsamı

**Senkronizasyon fonksiyonları** (`workgroupBarrier`, `storageBarrier`, `textureBarrier`): Execution ve memory scope olarak `Workgroup` kullanır.

#### 14.5.4 Memory Semantics

- Tüm **atomik** yerleşik fonksiyonlar `Relaxed` memory semantics kullanır (storage class semantics yok).
- **`workgroupBarrier`:** `AcquireRelease` sıralama + `WorkgroupMemory` semantics
- **`storageBarrier`:** `AcquireRelease` sıralama + `UniformMemory` semantics
- **`textureBarrier`:** `AcquireRelease` sıralama + `ImageMemory` semantics

> **Not:** WGSL'de adres uzayı (address space) kavramı, SPIR-V'deki storage class'a eşdeğerdir.

#### 14.5.5 Private vs Non-private

`storage`, `workgroup` ve `handle` adres uzaylarındaki tüm atomik olmayan okuma/yazma erişimleri **non-private** kabul edilir:

| Adres Uzayı | Okuma Operandı | Yazma Operandı |
|-------------|----------------|----------------|
| `storage` / `workgroup` | `NonPrivatePointer \| MakePointerVisible` (Workgroup scope) | `NonPrivatePointer \| MakePointerAvailable` (Workgroup scope) |
| `handle` | `NonPrivateTexel \| MakeTexelVisible` (Workgroup scope) | `NonPrivateTexel \| MakeTexelAvailable` (Workgroup scope) |

Bu, paylaşılan bellekteki değişikliklerin uygun senkronizasyon bariyerleri kullanıldığında diğer invocation'lara görünür olmasını sağlar.

---

> **Önceki:** [← Program Akışı ve Fonksiyonlar](04-program-akisi-ve-fonksiyonlar.md) · **Sonraki:** [Paralel Çalışma ve Doğruluk →](06-paralel-calisma-ve-dogruluk.md)
