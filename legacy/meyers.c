/*
 * Headers
*/

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <cuda.h>
#include <nvrtc.h>


/*
 * Initialisation
*/

int futhark_get_num_sizes(void);
const char *futhark_get_size_name(int);
const char *futhark_get_size_class(int);
struct futhark_context_config ;
struct futhark_context_config *futhark_context_config_new(void);
void futhark_context_config_free(struct futhark_context_config *cfg);
void futhark_context_config_add_nvrtc_option(struct futhark_context_config *cfg,
                                             const char *opt);
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag);
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag);
void futhark_context_config_set_device(struct futhark_context_config *cfg, const
                                       char *s);
void futhark_context_config_dump_program_to(struct futhark_context_config *cfg,
                                            const char *path);
void
futhark_context_config_load_program_from(struct futhark_context_config *cfg,
                                         const char *path);
void futhark_context_config_dump_ptx_to(struct futhark_context_config *cfg,
                                        const char *path);
void futhark_context_config_load_ptx_from(struct futhark_context_config *cfg,
                                          const char *path);
void
futhark_context_config_set_default_block_size(struct futhark_context_config *cfg,
                                              int size);
void
futhark_context_config_set_default_grid_size(struct futhark_context_config *cfg,
                                             int num);
void
futhark_context_config_set_default_tile_size(struct futhark_context_config *cfg,
                                             int num);
void
futhark_context_config_set_default_threshold(struct futhark_context_config *cfg,
                                             int num);
int futhark_context_config_set_size(struct futhark_context_config *cfg, const
                                    char *size_name, size_t size_value);
struct futhark_context ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg);
void futhark_context_free(struct futhark_context *ctx);
int futhark_context_sync(struct futhark_context *ctx);
char *futhark_context_get_error(struct futhark_context *ctx);

/*
 * Arrays
*/

struct futhark_i32_1d ;
struct futhark_i32_1d *futhark_new_i32_1d(struct futhark_context *ctx,
                                          int32_t *data, int dim0);
struct futhark_i32_1d *futhark_new_raw_i32_1d(struct futhark_context *ctx,
                                              CUdeviceptr data, int offset,
                                              int dim0);
int futhark_free_i32_1d(struct futhark_context *ctx,
                        struct futhark_i32_1d *arr);
int futhark_values_i32_1d(struct futhark_context *ctx,
                          struct futhark_i32_1d *arr, int32_t *data);
CUdeviceptr futhark_values_raw_i32_1d(struct futhark_context *ctx,
                                      struct futhark_i32_1d *arr);
int64_t *futhark_shape_i32_1d(struct futhark_context *ctx,
                              struct futhark_i32_1d *arr);
struct futhark_u16_1d ;
struct futhark_u16_1d *futhark_new_u16_1d(struct futhark_context *ctx,
                                          uint16_t *data, int dim0);
struct futhark_u16_1d *futhark_new_raw_u16_1d(struct futhark_context *ctx,
                                              CUdeviceptr data, int offset,
                                              int dim0);
int futhark_free_u16_1d(struct futhark_context *ctx,
                        struct futhark_u16_1d *arr);
int futhark_values_u16_1d(struct futhark_context *ctx,
                          struct futhark_u16_1d *arr, uint16_t *data);
CUdeviceptr futhark_values_raw_u16_1d(struct futhark_context *ctx,
                                      struct futhark_u16_1d *arr);
int64_t *futhark_shape_u16_1d(struct futhark_context *ctx,
                              struct futhark_u16_1d *arr);
struct futhark_u64_1d ;
struct futhark_u64_1d *futhark_new_u64_1d(struct futhark_context *ctx,
                                          uint64_t *data, int dim0);
struct futhark_u64_1d *futhark_new_raw_u64_1d(struct futhark_context *ctx,
                                              CUdeviceptr data, int offset,
                                              int dim0);
int futhark_free_u64_1d(struct futhark_context *ctx,
                        struct futhark_u64_1d *arr);
int futhark_values_u64_1d(struct futhark_context *ctx,
                          struct futhark_u64_1d *arr, uint64_t *data);
CUdeviceptr futhark_values_raw_u64_1d(struct futhark_context *ctx,
                                      struct futhark_u64_1d *arr);
int64_t *futhark_shape_u64_1d(struct futhark_context *ctx,
                              struct futhark_u64_1d *arr);

/*
 * Opaque values
*/


/*
 * Entry points
*/

int futhark_entry_main(struct futhark_context *ctx,
                       struct futhark_i32_1d **out0,
                       struct futhark_u64_1d **out1,
                       struct futhark_u64_1d **out2, const
                       struct futhark_u16_1d *in0, const
                       struct futhark_u16_1d *in1, const int32_t in2);

/*
 * Miscellaneous
*/

void futhark_debugging_report(struct futhark_context *ctx);
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
#undef NDEBUG
#include <assert.h>
/* Crash and burn. */

#include <stdarg.h>

static const char *fut_progname;

static void panic(int eval, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
        fprintf(stderr, "%s: ", fut_progname);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
        exit(eval);
}

/* For generating arbitrary-sized error messages.  It is the callers
   responsibility to free the buffer at some point. */
static char* msgprintf(const char *s, ...) {
  va_list vl;
  va_start(vl, s);
  size_t needed = 1 + vsnprintf(NULL, 0, s, vl);
  char *buffer = malloc(needed);
  va_start(vl, s); /* Must re-init. */
  vsnprintf(buffer, needed, s, vl);
  return buffer;
}

/* Some simple utilities for wall-clock timing.

   The function get_wall_time() returns the wall time in microseconds
   (with an unspecified offset).
*/

#ifdef _WIN32

#include <windows.h>

static int64_t get_wall_time(void) {
  LARGE_INTEGER time,freq;
  assert(QueryPerformanceFrequency(&freq));
  assert(QueryPerformanceCounter(&time));
  return ((double)time.QuadPart / freq.QuadPart) * 1000000;
}

#else
/* Assuming POSIX */

#include <time.h>
#include <sys/time.h>

static int64_t get_wall_time(void) {
  struct timeval time;
  assert(gettimeofday(&time,NULL) == 0);
  return time.tv_sec * 1000000 + time.tv_usec;
}

#endif

#include <string.h>
#include <inttypes.h>
#include <errno.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
//// Text I/O

typedef int (*writer)(FILE*, void*);
typedef int (*bin_reader)(void*);
typedef int (*str_reader)(const char *, void*);

struct array_reader {
  char* elems;
  int64_t n_elems_space;
  int64_t elem_size;
  int64_t n_elems_used;
  int64_t *shape;
  str_reader elem_reader;
};

static void skipspaces() {
  int c;
  do {
    c = getchar();
  } while (isspace(c));

  if (c != EOF) {
    ungetc(c, stdin);
  }
}

static int constituent(char c) {
  return isalnum(c) || c == '.' || c == '-' || c == '+' || c == '_';
}

// Produces an empty token only on EOF.
static void next_token(char *buf, int bufsize) {
 start:
  skipspaces();

  int i = 0;
  while (i < bufsize) {
    int c = getchar();
    buf[i] = c;

    if (c == EOF) {
      buf[i] = 0;
      return;
    } else if (c == '-' && i == 1 && buf[0] == '-') {
      // Line comment, so skip to end of line and start over.
      for (; c != '\n' && c != EOF; c = getchar());
      goto start;
    } else if (!constituent(c)) {
      if (i == 0) {
        // We permit single-character tokens that are not
        // constituents; this lets things like ']' and ',' be
        // tokens.
        buf[i+1] = 0;
        return;
      } else {
        ungetc(c, stdin);
        buf[i] = 0;
        return;
      }
    }

    i++;
  }

  buf[bufsize-1] = 0;
}

static int next_token_is(char *buf, int bufsize, const char* expected) {
  next_token(buf, bufsize);
  return strcmp(buf, expected) == 0;
}

static void remove_underscores(char *buf) {
  char *w = buf;

  for (char *r = buf; *r; r++) {
    if (*r != '_') {
      *w++ = *r;
    }
  }

  *w++ = 0;
}

static int read_str_elem(char *buf, struct array_reader *reader) {
  int ret;
  if (reader->n_elems_used == reader->n_elems_space) {
    reader->n_elems_space *= 2;
    reader->elems = (char*) realloc(reader->elems,
                                    reader->n_elems_space * reader->elem_size);
  }

  ret = reader->elem_reader(buf, reader->elems + reader->n_elems_used * reader->elem_size);

  if (ret == 0) {
    reader->n_elems_used++;
  }

  return ret;
}

static int read_str_array_elems(char *buf, int bufsize,
                                struct array_reader *reader, int dims) {
  int ret;
  int first = 1;
  char *knows_dimsize = (char*) calloc(dims,sizeof(char));
  int cur_dim = dims-1;
  int64_t *elems_read_in_dim = (int64_t*) calloc(dims,sizeof(int64_t));

  while (1) {
    next_token(buf, bufsize);

    if (strcmp(buf, "]") == 0) {
      if (knows_dimsize[cur_dim]) {
        if (reader->shape[cur_dim] != elems_read_in_dim[cur_dim]) {
          ret = 1;
          break;
        }
      } else {
        knows_dimsize[cur_dim] = 1;
        reader->shape[cur_dim] = elems_read_in_dim[cur_dim];
      }
      if (cur_dim == 0) {
        ret = 0;
        break;
      } else {
        cur_dim--;
        elems_read_in_dim[cur_dim]++;
      }
    } else if (strcmp(buf, ",") == 0) {
      next_token(buf, bufsize);
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        first = 1;
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else if (cur_dim == dims - 1) {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
      } else {
        ret = 1;
        break;
      }
    } else if (strlen(buf) == 0) {
      // EOF
      ret = 1;
      break;
    } else if (first) {
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
        first = 0;
      }
    } else {
      ret = 1;
      break;
    }
  }

  free(knows_dimsize);
  free(elems_read_in_dim);
  return ret;
}

static int read_str_empty_array(char *buf, int bufsize,
                                const char *type_name, int64_t *shape, int64_t dims) {
  if (strlen(buf) == 0) {
    // EOF
    return 1;
  }

  if (strcmp(buf, "empty") != 0) {
    return 1;
  }

  if (!next_token_is(buf, bufsize, "(")) {
    return 1;
  }

  for (int i = 0; i < dims-1; i++) {
    if (!next_token_is(buf, bufsize, "[")) {
      return 1;
    }

    if (!next_token_is(buf, bufsize, "]")) {
      return 1;
    }
  }

  if (!next_token_is(buf, bufsize, type_name)) {
    return 1;
  }


  if (!next_token_is(buf, bufsize, ")")) {
    return 1;
  }

  for (int i = 0; i < dims; i++) {
    shape[i] = 0;
  }

  return 0;
}

static int read_str_array(int64_t elem_size, str_reader elem_reader,
                          const char *type_name,
                          void **data, int64_t *shape, int64_t dims) {
  int ret;
  struct array_reader reader;
  char buf[100];

  int dims_seen;
  for (dims_seen = 0; dims_seen < dims; dims_seen++) {
    if (!next_token_is(buf, sizeof(buf), "[")) {
      break;
    }
  }

  if (dims_seen == 0) {
    return read_str_empty_array(buf, sizeof(buf), type_name, shape, dims);
  }

  if (dims_seen != dims) {
    return 1;
  }

  reader.shape = shape;
  reader.n_elems_used = 0;
  reader.elem_size = elem_size;
  reader.n_elems_space = 16;
  reader.elems = (char*) realloc(*data, elem_size*reader.n_elems_space);
  reader.elem_reader = elem_reader;

  ret = read_str_array_elems(buf, sizeof(buf), &reader, dims);

  *data = reader.elems;

  return ret;
}

#define READ_STR(MACRO, PTR, SUFFIX)                                   \
  remove_underscores(buf);                                              \
  int j;                                                                \
  if (sscanf(buf, "%"MACRO"%n", (PTR*)dest, &j) == 1) {                 \
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, SUFFIX) == 0);     \
  } else {                                                              \
    return 1;                                                           \
  }

static int read_str_i8(char *buf, void* dest) {
  /* Some platforms (WINDOWS) does not support scanf %hhd or its
     cousin, %SCNi8.  Read into int first to avoid corrupting
     memory.

     https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417  */
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(int8_t*)dest = x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "i8") == 0);
  } else {
    return 1;
  }
}

static int read_str_u8(char *buf, void* dest) {
  /* Some platforms (WINDOWS) does not support scanf %hhd or its
     cousin, %SCNu8.  Read into int first to avoid corrupting
     memory.

     https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417  */
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(uint8_t*)dest = x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "u8") == 0);
  } else {
    return 1;
  }
}

static int read_str_i16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "i16");
}

static int read_str_u16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "u16");
}

static int read_str_i32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "i32");
}

static int read_str_u32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "u32");
}

static int read_str_i64(char *buf, void* dest) {
  READ_STR(SCNi64, int64_t, "i64");
}

static int read_str_u64(char *buf, void* dest) {
  // FIXME: This is not correct, as SCNu64 only permits decimal
  // literals.  However, SCNi64 does not handle very large numbers
  // correctly (it's really for signed numbers, so that's fair).
  READ_STR(SCNu64, uint64_t, "u64");
}

static int read_str_f32(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f32.nan") == 0) {
    *(float*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f32.inf") == 0) {
    *(float*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f32.inf") == 0) {
    *(float*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("f", float, "f32");
  }
}

static int read_str_f64(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f64.nan") == 0) {
    *(double*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f64.inf") == 0) {
    *(double*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f64.inf") == 0) {
    *(double*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("lf", double, "f64");
  }
}

static int read_str_bool(char *buf, void* dest) {
  if (strcmp(buf, "true") == 0) {
    *(char*)dest = 1;
    return 0;
  } else if (strcmp(buf, "false") == 0) {
    *(char*)dest = 0;
    return 0;
  } else {
    return 1;
  }
}

static int write_str_i8(FILE *out, int8_t *src) {
  return fprintf(out, "%hhdi8", *src);
}

static int write_str_u8(FILE *out, uint8_t *src) {
  return fprintf(out, "%hhuu8", *src);
}

static int write_str_i16(FILE *out, int16_t *src) {
  return fprintf(out, "%hdi16", *src);
}

static int write_str_u16(FILE *out, uint16_t *src) {
  return fprintf(out, "%huu16", *src);
}

static int write_str_i32(FILE *out, int32_t *src) {
  return fprintf(out, "%di32", *src);
}

static int write_str_u32(FILE *out, uint32_t *src) {
  return fprintf(out, "%uu32", *src);
}

static int write_str_i64(FILE *out, int64_t *src) {
  return fprintf(out, "%"PRIi64"i64", *src);
}

static int write_str_u64(FILE *out, uint64_t *src) {
  return fprintf(out, "%"PRIu64"u64", *src);
}

static int write_str_f32(FILE *out, float *src) {
  float x = *src;
  if (isnan(x)) {
    return fprintf(out, "f32.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f32.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f32.inf");
  } else {
    return fprintf(out, "%.6ff32", x);
  }
}

static int write_str_f64(FILE *out, double *src) {
  double x = *src;
  if (isnan(x)) {
    return fprintf(out, "f64.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f64.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f64.inf");
  } else {
    return fprintf(out, "%.6ff64", *src);
  }
}

static int write_str_bool(FILE *out, void *src) {
  return fprintf(out, *(char*)src ? "true" : "false");
}

//// Binary I/O

#define BINARY_FORMAT_VERSION 2
#define IS_BIG_ENDIAN (!*(unsigned char *)&(uint16_t){1})

// On Windows we need to explicitly set the file mode to not mangle
// newline characters.  On *nix there is no difference.
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
static void set_binary_mode(FILE *f) {
  setmode(fileno(f), O_BINARY);
}
#else
static void set_binary_mode(FILE *f) {
  (void)f;
}
#endif

// Reading little-endian byte sequences.  On big-endian hosts, we flip
// the resulting bytes.

static int read_byte(void* dest) {
  int num_elems_read = fread(dest, 1, 1, stdin);
  return num_elems_read == 1 ? 0 : 1;
}

static int read_le_2byte(void* dest) {
  uint16_t x;
  int num_elems_read = fread(&x, 2, 1, stdin);
  if (IS_BIG_ENDIAN) {
    x = (x>>8) | (x<<8);
  }
  *(uint16_t*)dest = x;
  return num_elems_read == 1 ? 0 : 1;
}

static int read_le_4byte(void* dest) {
  uint32_t x;
  int num_elems_read = fread(&x, 4, 1, stdin);
  if (IS_BIG_ENDIAN) {
    x =
      ((x>>24)&0xFF) |
      ((x>>8) &0xFF00) |
      ((x<<8) &0xFF0000) |
      ((x<<24)&0xFF000000);
  }
  *(uint32_t*)dest = x;
  return num_elems_read == 1 ? 0 : 1;
}

static int read_le_8byte(void* dest) {
  uint64_t x;
  int num_elems_read = fread(&x, 8, 1, stdin);
  if (IS_BIG_ENDIAN) {
    x =
      ((x>>56)&0xFFull) |
      ((x>>40)&0xFF00ull) |
      ((x>>24)&0xFF0000ull) |
      ((x>>8) &0xFF000000ull) |
      ((x<<8) &0xFF00000000ull) |
      ((x<<24)&0xFF0000000000ull) |
      ((x<<40)&0xFF000000000000ull) |
      ((x<<56)&0xFF00000000000000ull);
  }
  *(uint64_t*)dest = x;
  return num_elems_read == 1 ? 0 : 1;
}

static int write_byte(void* dest) {
  int num_elems_written = fwrite(dest, 1, 1, stdin);
  return num_elems_written == 1 ? 0 : 1;
}

static int write_le_2byte(void* dest) {
  uint16_t x = *(uint16_t*)dest;
  if (IS_BIG_ENDIAN) {
    x = (x>>8) | (x<<8);
  }
  int num_elems_written = fwrite(&x, 2, 1, stdin);
  return num_elems_written == 1 ? 0 : 1;
}

static int write_le_4byte(void* dest) {
  uint32_t x = *(uint32_t*)dest;
  if (IS_BIG_ENDIAN) {
    x =
      ((x>>24)&0xFF) |
      ((x>>8) &0xFF00) |
      ((x<<8) &0xFF0000) |
      ((x<<24)&0xFF000000);
  }
  int num_elems_written = fwrite(&x, 4, 1, stdin);
  return num_elems_written == 1 ? 0 : 1;
}

static int write_le_8byte(void* dest) {
  uint64_t x = *(uint64_t*)dest;
  if (IS_BIG_ENDIAN) {
    x =
      ((x>>56)&0xFFull) |
      ((x>>40)&0xFF00ull) |
      ((x>>24)&0xFF0000ull) |
      ((x>>8) &0xFF000000ull) |
      ((x<<8) &0xFF00000000ull) |
      ((x<<24)&0xFF0000000000ull) |
      ((x<<40)&0xFF000000000000ull) |
      ((x<<56)&0xFF00000000000000ull);
  }
  int num_elems_written = fwrite(&x, 8, 1, stdin);
  return num_elems_written == 1 ? 0 : 1;
}

//// Types

struct primtype_info_t {
  const char binname[4]; // Used for parsing binary data.
  const char* type_name; // Same name as in Futhark.
  const int size; // in bytes
  const writer write_str; // Write in text format.
  const str_reader read_str; // Read in text format.
  const writer write_bin; // Write in binary format.
  const bin_reader read_bin; // Read in binary format.
};

static const struct primtype_info_t i8_info =
  {.binname = "  i8", .type_name = "i8",   .size = 1,
   .write_str = (writer)write_str_i8, .read_str = (str_reader)read_str_i8,
   .write_bin = (writer)write_byte, .read_bin = (bin_reader)read_byte};
static const struct primtype_info_t i16_info =
  {.binname = " i16", .type_name = "i16",  .size = 2,
   .write_str = (writer)write_str_i16, .read_str = (str_reader)read_str_i16,
   .write_bin = (writer)write_le_2byte, .read_bin = (bin_reader)read_le_2byte};
static const struct primtype_info_t i32_info =
  {.binname = " i32", .type_name = "i32",  .size = 4,
   .write_str = (writer)write_str_i32, .read_str = (str_reader)read_str_i32,
   .write_bin = (writer)write_le_4byte, .read_bin = (bin_reader)read_le_4byte};
static const struct primtype_info_t i64_info =
  {.binname = " i64", .type_name = "i64",  .size = 8,
   .write_str = (writer)write_str_i64, .read_str = (str_reader)read_str_i64,
   .write_bin = (writer)write_le_8byte, .read_bin = (bin_reader)read_le_8byte};
static const struct primtype_info_t u8_info =
  {.binname = "  u8", .type_name = "u8",   .size = 1,
   .write_str = (writer)write_str_u8, .read_str = (str_reader)read_str_u8,
   .write_bin = (writer)write_byte, .read_bin = (bin_reader)read_byte};
static const struct primtype_info_t u16_info =
  {.binname = " u16", .type_name = "u16",  .size = 2,
   .write_str = (writer)write_str_u16, .read_str = (str_reader)read_str_u16,
   .write_bin = (writer)write_le_2byte, .read_bin = (bin_reader)read_le_2byte};
static const struct primtype_info_t u32_info =
  {.binname = " u32", .type_name = "u32",  .size = 4,
   .write_str = (writer)write_str_u32, .read_str = (str_reader)read_str_u32,
   .write_bin = (writer)write_le_4byte, .read_bin = (bin_reader)read_le_4byte};
static const struct primtype_info_t u64_info =
  {.binname = " u64", .type_name = "u64",  .size = 8,
   .write_str = (writer)write_str_u64, .read_str = (str_reader)read_str_u64,
   .write_bin = (writer)write_le_8byte, .read_bin = (bin_reader)read_le_8byte};
static const struct primtype_info_t f32_info =
  {.binname = " f32", .type_name = "f32",  .size = 4,
   .write_str = (writer)write_str_f32, .read_str = (str_reader)read_str_f32,
   .write_bin = (writer)write_le_4byte, .read_bin = (bin_reader)read_le_4byte};
static const struct primtype_info_t f64_info =
  {.binname = " f64", .type_name = "f64",  .size = 8,
   .write_str = (writer)write_str_f64, .read_str = (str_reader)read_str_f64,
   .write_bin = (writer)write_le_8byte, .read_bin = (bin_reader)read_le_8byte};
static const struct primtype_info_t bool_info =
  {.binname = "bool", .type_name = "bool", .size = 1,
   .write_str = (writer)write_str_bool, .read_str = (str_reader)read_str_bool,
   .write_bin = (writer)write_byte, .read_bin = (bin_reader)read_byte};

static const struct primtype_info_t* primtypes[] = {
  &i8_info, &i16_info, &i32_info, &i64_info,
  &u8_info, &u16_info, &u32_info, &u64_info,
  &f32_info, &f64_info,
  &bool_info,
  NULL // NULL-terminated
};

// General value interface.  All endian business taken care of at
// lower layers.

static int read_is_binary() {
  skipspaces();
  int c = getchar();
  if (c == 'b') {
    int8_t bin_version;
    int ret = read_byte(&bin_version);

    if (ret != 0) { panic(1, "binary-input: could not read version.\n"); }

    if (bin_version != BINARY_FORMAT_VERSION) {
      panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
            bin_version, BINARY_FORMAT_VERSION);
    }

    return 1;
  }
  ungetc(c, stdin);
  return 0;
}

static const struct primtype_info_t* read_bin_read_type_enum() {
  char read_binname[4];

  int num_matched = scanf("%4c", read_binname);
  if (num_matched != 1) { panic(1, "binary-input: Couldn't read element type.\n"); }

  const struct primtype_info_t **type = primtypes;

  for (; *type != NULL; type++) {
    // I compare the 4 characters manually instead of using strncmp because
    // this allows any value to be used, also NULL bytes
    if (memcmp(read_binname, (*type)->binname, 4) == 0) {
      return *type;
    }
  }
  panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname);
  return NULL;
}

static void read_bin_ensure_scalar(const struct primtype_info_t *expected_type) {
  int8_t bin_dims;
  int ret = read_byte(&bin_dims);
  if (ret != 0) { panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != 0) {
    panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n",
          bin_dims);
  }

  const struct primtype_info_t *bin_type = read_bin_read_type_enum();
  if (bin_type != expected_type) {
    panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
          expected_type->type_name,
          bin_type->type_name);
  }
}

//// High-level interface

static int read_bin_array(const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  int ret;

  int8_t bin_dims;
  ret = read_byte(&bin_dims);
  if (ret != 0) { panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != dims) {
    panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
          dims, bin_dims);
  }

  const struct primtype_info_t *bin_primtype = read_bin_read_type_enum();
  if (expected_type != bin_primtype) {
    panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
          dims, expected_type->type_name, dims, bin_primtype->type_name);
  }

  uint64_t elem_count = 1;
  for (int i=0; i<dims; i++) {
    uint64_t bin_shape;
    ret = read_le_8byte(&bin_shape);
    if (ret != 0) { panic(1, "binary-input: Couldn't read size for dimension %i of array.\n", i); }
    elem_count *= bin_shape;
    shape[i] = (int64_t) bin_shape;
  }

  size_t elem_size = expected_type->size;
  void* tmp = realloc(*data, elem_count * elem_size);
  if (tmp == NULL) {
    panic(1, "binary-input: Failed to allocate array of size %i.\n",
          elem_count * elem_size);
  }
  *data = tmp;

  size_t num_elems_read = fread(*data, elem_size, elem_count, stdin);
  if (num_elems_read != elem_count) {
    panic(1, "binary-input: tried to read %i elements of an array, but only got %i elements.\n",
          elem_count, num_elems_read);
  }

  // If we're on big endian platform we must change all multibyte elements
  // from using little endian to big endian
  if (IS_BIG_ENDIAN && elem_size != 1) {
    char* elems = (char*) *data;
    for (uint64_t i=0; i<elem_count; i++) {
      char* elem = elems+(i*elem_size);
      for (unsigned int j=0; j<elem_size/2; j++) {
        char head = elem[j];
        int tail_index = elem_size-1-j;
        elem[j] = elem[tail_index];
        elem[tail_index] = head;
      }
    }
  }

  return 0;
}

static int read_array(const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  if (!read_is_binary()) {
    return read_str_array(expected_type->size, (str_reader)expected_type->read_str, expected_type->type_name, data, shape, dims);
  } else {
    return read_bin_array(expected_type, data, shape, dims);
  }
}

static int write_str_array(FILE *out, const struct primtype_info_t *elem_type, unsigned char *data, int64_t *shape, int8_t rank) {
  if (rank==0) {
    elem_type->write_str(out, (void*)data);
  } else {
    int64_t len = shape[0];
    int64_t slice_size = 1;

    int64_t elem_size = elem_type->size;
    for (int64_t i = 1; i < rank; i++) {
      slice_size *= shape[i];
    }

    if (len*slice_size == 0) {
      printf("empty(");
      for (int64_t i = 1; i < rank; i++) {
        printf("[]");
      }
      printf("%s", elem_type->type_name);
      printf(")");
    } else if (rank==1) {
      putchar('[');
      for (int64_t i = 0; i < len; i++) {
        elem_type->write_str(out, (void*) (data + i * elem_size));
        if (i != len-1) {
          printf(", ");
        }
      }
      putchar(']');
    } else {
      putchar('[');
      for (int64_t i = 0; i < len; i++) {
        write_str_array(out, elem_type, data + i * slice_size * elem_size, shape+1, rank-1);
        if (i != len-1) {
          printf(", ");
        }
      }
      putchar(']');
    }
  }
  return 0;
}

static int write_bin_array(FILE *out, const struct primtype_info_t *elem_type, unsigned char *data, int64_t *shape, int8_t rank) {
  int64_t num_elems = 1;
  for (int64_t i = 0; i < rank; i++) {
    num_elems *= shape[i];
  }

  fputc('b', out);
  fputc((char)BINARY_FORMAT_VERSION, out);
  fwrite(&rank, sizeof(int8_t), 1, out);
  fputs(elem_type->binname, out);
  fwrite(shape, sizeof(int64_t), rank, out);

  if (IS_BIG_ENDIAN) {
    for (int64_t i = 0; i < num_elems; i++) {
      unsigned char *elem = data+i*elem_type->size;
      for (int64_t j = 0; j < elem_type->size; j++) {
        fwrite(&elem[elem_type->size-j], 1, 1, out);
      }
    }
  } else {
    fwrite(data, elem_type->size, num_elems, out);
  }

  return 0;
}

static int write_array(FILE *out, int write_binary,
                       const struct primtype_info_t *elem_type, void *data, int64_t *shape, int8_t rank) {
  if (write_binary) {
    return write_bin_array(out, elem_type, data, shape, rank);
  } else {
    return write_str_array(out, elem_type, data, shape, rank);
  }
}

static int read_scalar(const struct primtype_info_t *expected_type, void *dest) {
  if (!read_is_binary()) {
    char buf[100];
    next_token(buf, sizeof(buf));
    return expected_type->read_str(buf, dest);
  } else {
    read_bin_ensure_scalar(expected_type);
    return expected_type->read_bin(dest);
  }
}

static int write_scalar(FILE *out, int write_binary, const struct primtype_info_t *type, void *src) {
  if (write_binary) {
    return write_bin_array(out, type, src, NULL, 0);
  } else {
    return type->write_str(out, src);
  }
}

static int binary_output = 0;
static FILE *runtime_file;
static int perform_warmup = 0;
static int num_runs = 1;
static const char *entry_point = "main";
// Start of tuning.h

static char* load_tuning_file(const char *fname,
                              void *cfg,
                              int (*set_size)(void*, const char*, size_t)) {
  const int max_line_len = 1024;
  char* line = malloc(max_line_len);

  FILE *f = fopen(fname, "r");

  if (f == NULL) {
    snprintf(line, max_line_len, "Cannot open file: %s", strerror(errno));
    return line;
  }

  int lineno = 0;
  while (fgets(line, max_line_len, f) != NULL) {
    lineno++;
    char *eql = strstr(line, "=");
    if (eql) {
      *eql = 0;
      int value = atoi(eql+1);
      if (set_size(cfg, line, value) != 0) {
        strncpy(eql+1, line, max_line_len-strlen(line)-1);
        snprintf(line, max_line_len, "Unknown name '%s' on line %d.", eql+1, lineno);
        return line;
      }
    } else {
      snprintf(line, max_line_len, "Invalid line %d (must be of form 'name=int').",
               lineno);
      return line;
    }
  }

  free(line);

  return NULL;
}

// End of tuning.h

int parse_options(struct futhark_context_config *cfg, int argc,
                  char *const argv[])
{
    int ch;
    static struct option long_options[] = {{"write-runtime-to",
                                            required_argument, NULL, 1},
                                           {"runs", required_argument, NULL, 2},
                                           {"debugging", no_argument, NULL, 3},
                                           {"log", no_argument, NULL, 4},
                                           {"entry-point", required_argument,
                                            NULL, 5}, {"binary-output",
                                                       no_argument, NULL, 6},
                                           {"dump-cuda", required_argument,
                                            NULL, 7}, {"load-cuda",
                                                       required_argument, NULL,
                                                       8}, {"dump-ptx",
                                                            required_argument,
                                                            NULL, 9},
                                           {"load-ptx", required_argument, NULL,
                                            10}, {"nvrtc-option",
                                                  required_argument, NULL, 11},
                                           {"print-sizes", no_argument, NULL,
                                            12}, {"tuning", required_argument,
                                                  NULL, 13}, {0, 0, 0, 0}};
    
    while ((ch = getopt_long(argc, argv, ":t:r:DLe:b", long_options, NULL)) !=
           -1) {
        if (ch == 1 || ch == 't') {
            runtime_file = fopen(optarg, "w");
            if (runtime_file == NULL)
                panic(1, "Cannot open %s: %s\n", optarg, strerror(errno));
        }
        if (ch == 2 || ch == 'r') {
            num_runs = atoi(optarg);
            perform_warmup = 1;
            if (num_runs <= 0)
                panic(1, "Need a positive number of runs, not %s\n", optarg);
        }
        if (ch == 3 || ch == 'D')
            futhark_context_config_set_debugging(cfg, 1);
        if (ch == 4 || ch == 'L')
            futhark_context_config_set_logging(cfg, 1);
        if (ch == 5 || ch == 'e') {
            if (entry_point != NULL)
                entry_point = optarg;
        }
        if (ch == 6 || ch == 'b')
            binary_output = 1;
        if (ch == 7) {
            futhark_context_config_dump_program_to(cfg, optarg);
            entry_point = NULL;
        }
        if (ch == 8)
            futhark_context_config_load_program_from(cfg, optarg);
        if (ch == 9) {
            futhark_context_config_dump_ptx_to(cfg, optarg);
            entry_point = NULL;
        }
        if (ch == 10)
            futhark_context_config_load_ptx_from(cfg, optarg);
        if (ch == 11)
            futhark_context_config_add_nvrtc_option(cfg, optarg);
        if (ch == 12) {
            int n = futhark_get_num_sizes();
            
            for (int i = 0; i < n; i++)
                printf("%s (%s)\n", futhark_get_size_name(i),
                       futhark_get_size_class(i));
            exit(0);
        }
        if (ch == 13) {
            char *fname = optarg;
            char *ret = load_tuning_file(optarg, cfg, (int (*)(void *, const
                                                               char *,
                                                               size_t)) futhark_context_config_set_size);
            
            if (ret != NULL)
                panic(1, "When loading tuning from '%s': %s\n", optarg, ret);
        }
        if (ch == ':')
            panic(-1, "Missing argument for option %s\n", argv[optind - 1]);
        if (ch == '?') {
            fprintf(stderr, "Usage: %s: %s\n", fut_progname,
                    "[-t/--write-runtime-to FILE] [-r/--runs INT] [-D/--debugging] [-L/--log] [-e/--entry-point NAME] [-b/--binary-output] [--dump-cuda FILE] [--load-cuda FILE] [--dump-ptx FILE] [--load-ptx FILE] [--nvrtc-option OPT] [--print-sizes] [--tuning FILE]");
            panic(1, "Unknown option: %s\n", argv[optind - 1]);
        }
    }
    return optind;
}
static void futrts_cli_entry_main(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    struct futhark_u16_1d *read_value_7962;
    int64_t read_shape_7963[1];
    int16_t *read_arr_7964 = NULL;
    
    errno = 0;
    if (read_array(&u16_info, (void **) &read_arr_7964, read_shape_7963, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[]",
              u16_info.type_name, strerror(errno));
    
    struct futhark_u16_1d *read_value_7965;
    int64_t read_shape_7966[1];
    int16_t *read_arr_7967 = NULL;
    
    errno = 0;
    if (read_array(&u16_info, (void **) &read_arr_7967, read_shape_7966, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[]",
              u16_info.type_name, strerror(errno));
    
    int32_t read_value_7968;
    
    if (read_scalar(&i32_info, &read_value_7968) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 2,
              i32_info.type_name, strerror(errno));
    
    struct futhark_i32_1d *result_7969;
    struct futhark_u64_1d *result_7970;
    struct futhark_u64_1d *result_7971;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_7962 = futhark_new_u16_1d(ctx, read_arr_7964,
                                                     read_shape_7963[0])) != 0);
        assert((read_value_7965 = futhark_new_u16_1d(ctx, read_arr_7967,
                                                     read_shape_7966[0])) != 0);
        ;
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_7969, &result_7970, &result_7971,
                               read_value_7962, read_value_7965,
                               read_value_7968);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_u16_1d(ctx, read_value_7962) == 0);
        assert(futhark_free_u16_1d(ctx, read_value_7965) == 0);
        ;
        assert(futhark_free_i32_1d(ctx, result_7969) == 0);
        assert(futhark_free_u64_1d(ctx, result_7970) == 0);
        assert(futhark_free_u64_1d(ctx, result_7971) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_7962 = futhark_new_u16_1d(ctx, read_arr_7964,
                                                     read_shape_7963[0])) != 0);
        assert((read_value_7965 = futhark_new_u16_1d(ctx, read_arr_7967,
                                                     read_shape_7966[0])) != 0);
        ;
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_7969, &result_7970, &result_7971,
                               read_value_7962, read_value_7965,
                               read_value_7968);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_u16_1d(ctx, read_value_7962) == 0);
        assert(futhark_free_u16_1d(ctx, read_value_7965) == 0);
        ;
        if (run < num_runs - 1) {
            assert(futhark_free_i32_1d(ctx, result_7969) == 0);
            assert(futhark_free_u64_1d(ctx, result_7970) == 0);
            assert(futhark_free_u64_1d(ctx, result_7971) == 0);
        }
    }
    free(read_arr_7964);
    free(read_arr_7967);
    ;
    if (binary_output)
        set_binary_mode(stdout);
    {
        int32_t *arr = calloc(sizeof(int32_t), futhark_shape_i32_1d(ctx,
                                                                    result_7969)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_i32_1d(ctx, result_7969, arr) == 0);
        write_array(stdout, binary_output, &i32_info, arr,
                    futhark_shape_i32_1d(ctx, result_7969), 1);
        free(arr);
    }
    printf("\n");
    {
        int64_t *arr = calloc(sizeof(int64_t), futhark_shape_u64_1d(ctx,
                                                                    result_7970)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_u64_1d(ctx, result_7970, arr) == 0);
        write_array(stdout, binary_output, &u64_info, arr,
                    futhark_shape_u64_1d(ctx, result_7970), 1);
        free(arr);
    }
    printf("\n");
    {
        int64_t *arr = calloc(sizeof(int64_t), futhark_shape_u64_1d(ctx,
                                                                    result_7971)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_u64_1d(ctx, result_7971, arr) == 0);
        write_array(stdout, binary_output, &u64_info, arr,
                    futhark_shape_u64_1d(ctx, result_7971), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_i32_1d(ctx, result_7969) == 0);
    assert(futhark_free_u64_1d(ctx, result_7970) == 0);
    assert(futhark_free_u64_1d(ctx, result_7971) == 0);
}
typedef void entry_point_fun(struct futhark_context *);
struct entry_point_entry {
    const char *name;
    entry_point_fun *fun;
} ;
int main(int argc, char **argv)
{
    fut_progname = argv[0];
    
    struct entry_point_entry entry_points[] = {{.name ="main", .fun =
                                                futrts_cli_entry_main}};
    struct futhark_context_config *cfg = futhark_context_config_new();
    
    assert(cfg != NULL);
    
    int parsed_options = parse_options(cfg, argc, argv);
    
    argc -= parsed_options;
    argv += parsed_options;
    if (argc != 0)
        panic(1, "Excess non-option: %s\n", argv[0]);
    
    struct futhark_context *ctx = futhark_context_new(cfg);
    
    assert(ctx != NULL);
    if (entry_point != NULL) {
        int num_entry_points = sizeof(entry_points) / sizeof(entry_points[0]);
        entry_point_fun *entry_point_fun = NULL;
        
        for (int i = 0; i < num_entry_points; i++) {
            if (strcmp(entry_points[i].name, entry_point) == 0) {
                entry_point_fun = entry_points[i].fun;
                break;
            }
        }
        if (entry_point_fun == NULL) {
            fprintf(stderr,
                    "No entry point '%s'.  Select another with --entry-point.  Options are:\n",
                    entry_point);
            for (int i = 0; i < num_entry_points; i++)
                fprintf(stderr, "%s\n", entry_points[i].name);
            return 1;
        }
        entry_point_fun(ctx);
        if (runtime_file != NULL)
            fclose(runtime_file);
        futhark_debugging_report(ctx);
    }
    futhark_context_free(ctx);
    futhark_context_config_free(cfg);
    return 0;
}
#ifdef _MSC_VER
#define inline __inline
#endif
#include <string.h>
#include <inttypes.h>
#include <ctype.h>
#include <errno.h>
#include <assert.h>
/* A very simple cross-platform implementation of locks.  Uses
   pthreads on Unix and some Windows thing there.  Futhark's
   host-level code is not multithreaded, but user code may be, so we
   need some mechanism for ensuring atomic access to API functions.
   This is that mechanism.  It is not exposed to user code at all, so
   we do not have to worry about name collisions. */

#ifdef _WIN32

typedef HANDLE lock_t;

static lock_t create_lock(lock_t *lock) {
  *lock = CreateMutex(NULL,  /* Default security attributes. */
                      FALSE, /* Initially unlocked. */
                      NULL); /* Unnamed. */
}

static void lock_lock(lock_t *lock) {
  assert(WaitForSingleObject(*lock, INFINITE) == WAIT_OBJECT_0);
}

static void lock_unlock(lock_t *lock) {
  assert(ReleaseMutex(*lock));
}

static void free_lock(lock_t *lock) {
  CloseHandle(*lock);
}

#else
/* Assuming POSIX */

#include <pthread.h>

typedef pthread_mutex_t lock_t;

static void create_lock(lock_t *lock) {
  int r = pthread_mutex_init(lock, NULL);
  assert(r == 0);
}

static void lock_lock(lock_t *lock) {
  int r = pthread_mutex_lock(lock);
  assert(r == 0);
}

static void lock_unlock(lock_t *lock) {
  int r = pthread_mutex_unlock(lock);
  assert(r == 0);
}

static void free_lock(lock_t *lock) {
  /* Nothing to do for pthreads. */
  lock = lock;
}

#endif

#include <cuda.h>
#include <nvrtc.h>
typedef CUdeviceptr fl_mem_t;
/* Free list management */

/* An entry in the free list.  May be invalid, to avoid having to
   deallocate entries as soon as they are removed.  There is also a
   tag, to help with memory reuse. */
struct free_list_entry {
  size_t size;
  fl_mem_t mem;
  const char *tag;
  unsigned char valid;
};

struct free_list {
  struct free_list_entry *entries;        // Pointer to entries.
  int capacity;                           // Number of entries.
  int used;                               // Number of valid entries.
};

void free_list_init(struct free_list *l) {
  l->capacity = 30; // Picked arbitrarily.
  l->used = 0;
  l->entries = malloc(sizeof(struct free_list_entry) * l->capacity);
  for (int i = 0; i < l->capacity; i++) {
    l->entries[i].valid = 0;
  }
}

/* Remove invalid entries from the free list. */
void free_list_pack(struct free_list *l) {
  int p = 0;
  for (int i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid) {
      l->entries[p] = l->entries[i];
      p++;
    }
  }
  // Now p == l->used.
  l->entries = realloc(l->entries, l->used * sizeof(struct free_list_entry));
  l->capacity = l->used;
}

void free_list_destroy(struct free_list *l) {
  assert(l->used == 0);
  free(l->entries);
}

int free_list_find_invalid(struct free_list *l) {
  int i;
  for (i = 0; i < l->capacity; i++) {
    if (!l->entries[i].valid) {
      break;
    }
  }
  return i;
}

void free_list_insert(struct free_list *l, size_t size, fl_mem_t mem, const char *tag) {
  int i = free_list_find_invalid(l);

  if (i == l->capacity) {
    // List is full; so we have to grow it.
    int new_capacity = l->capacity * 2 * sizeof(struct free_list_entry);
    l->entries = realloc(l->entries, new_capacity);
    for (int j = 0; j < l->capacity; j++) {
      l->entries[j+l->capacity].valid = 0;
    }
    l->capacity *= 2;
  }

  // Now 'i' points to the first invalid entry.
  l->entries[i].valid = 1;
  l->entries[i].size = size;
  l->entries[i].mem = mem;
  l->entries[i].tag = tag;

  l->used++;
}

/* Find and remove a memory block of at least the desired size and
   tag.  Returns 0 on success.  */
int free_list_find(struct free_list *l, const char *tag, size_t *size_out, fl_mem_t *mem_out) {
  int i;
  for (i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid && l->entries[i].tag == tag) {
      l->entries[i].valid = 0;
      *size_out = l->entries[i].size;
      *mem_out = l->entries[i].mem;
      l->used--;
      return 0;
    }
  }

  return 1;
}

/* Remove the first block in the free list.  Returns 0 if a block was
   removed, and nonzero if the free list was already empty. */
int free_list_first(struct free_list *l, fl_mem_t *mem_out) {
  for (int i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid) {
      l->entries[i].valid = 0;
      *mem_out = l->entries[i].mem;
      l->used--;
      return 0;
    }
  }

  return 1;
}


/* Simple CUDA runtime framework */

#define CUDA_SUCCEED(x) cuda_api_succeed(x, #x, __FILE__, __LINE__)
#define NVRTC_SUCCEED(x) nvrtc_api_succeed(x, #x, __FILE__, __LINE__)

static inline void cuda_api_succeed(CUresult res, const char *call,
    const char *file, int line)
{
  if (res != CUDA_SUCCESS) {
    const char *err_str;
    cuGetErrorString(res, &err_str);
    if (err_str == NULL) { err_str = "Unknown"; }
    panic(-1, "%s:%d: CUDA call\n  %s\nfailed with error code %d (%s)\n",
        file, line, call, res, err_str);
  }
}

static inline void nvrtc_api_succeed(nvrtcResult res, const char *call,
    const char *file, int line)
{
  if (res != NVRTC_SUCCESS) {
    const char *err_str = nvrtcGetErrorString(res);
    panic(-1, "%s:%d: NVRTC call\n  %s\nfailed with error code %d (%s)\n",
        file, line, call, res, err_str);
  }
}

struct cuda_config {
  int debugging;
  int logging;
  const char *preferred_device;

  const char *dump_program_to;
  const char *load_program_from;

  const char *dump_ptx_to;
  const char *load_ptx_from;

  size_t default_block_size;
  size_t default_grid_size;
  size_t default_tile_size;
  size_t default_threshold;

  int default_block_size_changed;
  int default_grid_size_changed;
  int default_tile_size_changed;

  int num_sizes;
  const char **size_names;
  const char **size_vars;
  size_t *size_values;
  const char **size_classes;
};

void cuda_config_init(struct cuda_config *cfg,
                      int num_sizes,
                      const char *size_names[],
                      const char *size_vars[],
                      size_t *size_values,
                      const char *size_classes[])
{
  cfg->debugging = 0;
  cfg->logging = 0;
  cfg->preferred_device = "";

  cfg->dump_program_to = NULL;
  cfg->load_program_from = NULL;

  cfg->dump_ptx_to = NULL;
  cfg->load_ptx_from = NULL;

  cfg->default_block_size = 256;
  cfg->default_grid_size = 128;
  cfg->default_tile_size = 32;
  cfg->default_threshold = 32*1024;

  cfg->default_block_size_changed = 0;
  cfg->default_grid_size_changed = 0;
  cfg->default_tile_size_changed = 0;

  cfg->num_sizes = num_sizes;
  cfg->size_names = size_names;
  cfg->size_vars = size_vars;
  cfg->size_values = size_values;
  cfg->size_classes = size_classes;
}

struct cuda_context {
  CUdevice dev;
  CUcontext cu_ctx;
  CUmodule module;

  struct cuda_config cfg;

  struct free_list free_list;

  size_t max_block_size;
  size_t max_grid_size;
  size_t max_tile_size;
  size_t max_threshold;

  size_t lockstep_width;
};

#define CU_DEV_ATTR(x) (CU_DEVICE_ATTRIBUTE_##x)
#define device_query(dev,attrib) _device_query(dev, CU_DEV_ATTR(attrib))
static int _device_query(CUdevice dev, CUdevice_attribute attrib)
{
  int val;
  CUDA_SUCCEED(cuDeviceGetAttribute(&val, attrib, dev));
  return val;
}

#define CU_FUN_ATTR(x) (CU_FUNC_ATTRIBUTE_##x)
#define function_query(fn,attrib) _function_query(dev, CU_FUN_ATTR(attrib))
static int _function_query(CUfunction dev, CUfunction_attribute attrib)
{
  int val;
  CUDA_SUCCEED(cuFuncGetAttribute(&val, attrib, dev));
  return val;
}

void set_preferred_device(struct cuda_config *cfg, const char *s)
{
  cfg->preferred_device = s;
}

static int cuda_device_setup(struct cuda_context *ctx)
{
  char name[256];
  int count, chosen = -1, best_cc = -1;
  int cc_major_best, cc_minor_best;
  int cc_major, cc_minor;
  CUdevice dev;

  CUDA_SUCCEED(cuDeviceGetCount(&count));
  if (count == 0) { return 1; }

  // XXX: Current device selection policy is to choose the device with the
  // highest compute capability (if no preferred device is set).
  // This should maybe be changed, since greater compute capability is not
  // necessarily an indicator of better performance.
  for (int i = 0; i < count; i++) {
    CUDA_SUCCEED(cuDeviceGet(&dev, i));

    cc_major = device_query(dev, COMPUTE_CAPABILITY_MAJOR);
    cc_minor = device_query(dev, COMPUTE_CAPABILITY_MINOR);

    CUDA_SUCCEED(cuDeviceGetName(name, sizeof(name)/sizeof(name[0]) - 1, dev));
    name[sizeof(name)/sizeof(name[0])] = 0;

    if (ctx->cfg.debugging) {
      fprintf(stderr, "Device #%d: name=\"%s\", compute capability=%d.%d\n",
          i, name, cc_major, cc_minor);
    }

    if (device_query(dev, COMPUTE_MODE) == CU_COMPUTEMODE_PROHIBITED) {
      if (ctx->cfg.debugging) {
        fprintf(stderr, "Device #%d is compute-prohibited, ignoring\n", i);
      }
      continue;
    }

    if (best_cc == -1 || cc_major > cc_major_best ||
        (cc_major == cc_major_best && cc_minor > cc_minor_best)) {
      best_cc = i;
      cc_major_best = cc_major;
      cc_minor_best = cc_minor;
    }

    if (chosen == -1 && strstr(name, ctx->cfg.preferred_device) == name) {
      chosen = i;
    }
  }

  if (chosen == -1) { chosen = best_cc; }
  if (chosen == -1) { return 1; }

  if (ctx->cfg.debugging) {
    fprintf(stderr, "Using device #%d\n", chosen);
  }

  CUDA_SUCCEED(cuDeviceGet(&ctx->dev, chosen));
  return 0;
}

static char *concat_fragments(const char *src_fragments[])
{
  size_t src_len = 0;
  const char **p;

  for (p = src_fragments; *p; p++) {
    src_len += strlen(*p);
  }

  char *src = malloc(src_len + 1);
  size_t n = 0;
  for (p = src_fragments; *p; p++) {
    strcpy(src + n, *p);
    n += strlen(*p);
  }

  return src;
}

static const char *cuda_nvrtc_get_arch(CUdevice dev)
{
  struct {
    int major;
    int minor;
    const char *arch_str;
  } static const x[] = {
    { 3, 0, "compute_30" },
    { 3, 2, "compute_32" },
    { 3, 5, "compute_35" },
    { 3, 7, "compute_37" },
    { 5, 0, "compute_50" },
    { 5, 2, "compute_52" },
    { 5, 3, "compute_53" },
    { 6, 0, "compute_60" },
    { 6, 1, "compute_61" },
    { 6, 2, "compute_62" },
    { 7, 0, "compute_70" },
    { 7, 2, "compute_72" }
  };

  int major = device_query(dev, COMPUTE_CAPABILITY_MAJOR);
  int minor = device_query(dev, COMPUTE_CAPABILITY_MINOR);

  int chosen = -1;
  for (int i = 0; i < sizeof(x)/sizeof(x[0]); i++) {
    if (x[i].major < major || (x[i].major == major && x[i].minor <= minor)) {
      chosen = i;
    } else {
      break;
    }
  }

  if (chosen == -1) {
    panic(-1, "Unsupported compute capability %d.%d\n", major, minor);
  }
  return x[chosen].arch_str;
}

static char *cuda_nvrtc_build(struct cuda_context *ctx, const char *src,
                              const char *extra_opts[])
{
  nvrtcProgram prog;
  NVRTC_SUCCEED(nvrtcCreateProgram(&prog, src, "futhark-cuda", 0, NULL, NULL));
  int arch_set = 0, num_extra_opts;

  // nvrtc cannot handle multiple -arch options.  Hence, if one of the
  // extra_opts is -arch, we have to be careful not to do our usual
  // automatic generation.
  for (num_extra_opts = 0; extra_opts[num_extra_opts] != NULL; num_extra_opts++) {
    if (strstr(extra_opts[num_extra_opts], "-arch")
        == extra_opts[num_extra_opts] ||
        strstr(extra_opts[num_extra_opts], "--gpu-architecture")
        == extra_opts[num_extra_opts]) {
      arch_set = 1;
    }
  }

  size_t n_opts, i = 0, i_dyn, n_opts_alloc = 20 + num_extra_opts + ctx->cfg.num_sizes;
  const char **opts = malloc(n_opts_alloc * sizeof(const char *));
  if (!arch_set) {
    opts[i++] = "-arch";
    opts[i++] = cuda_nvrtc_get_arch(ctx->dev);
  }
  opts[i++] = "-default-device";
  if (ctx->cfg.debugging) {
    opts[i++] = "-G";
    opts[i++] = "-lineinfo";
  } else {
    opts[i++] = "--disable-warnings";
  }
  i_dyn = i;
  for (size_t j = 0; j < ctx->cfg.num_sizes; j++) {
    opts[i++] = msgprintf("-D%s=%zu", ctx->cfg.size_vars[j],
        ctx->cfg.size_values[j]);
  }
  opts[i++] = msgprintf("-DLOCKSTEP_WIDTH=%zu", ctx->lockstep_width);
  opts[i++] = msgprintf("-DMAX_THREADS_PER_BLOCK=%zu", ctx->max_block_size);

  // It is crucial that the extra_opts are last, so that the free()
  // logic below does not cause problems.
  for (int j = 0; extra_opts[j] != NULL; j++) {
    opts[i++] = extra_opts[j];
  }

  n_opts = i;

  if (ctx->cfg.debugging) {
    fprintf(stderr, "NVRTC compile options:\n");
    for (size_t j = 0; j < n_opts; j++) {
      fprintf(stderr, "\t%s\n", opts[j]);
    }
    fprintf(stderr, "\n");
  }

  nvrtcResult res = nvrtcCompileProgram(prog, n_opts, opts);
  if (res != NVRTC_SUCCESS) {
    size_t log_size;
    if (nvrtcGetProgramLogSize(prog, &log_size) == NVRTC_SUCCESS) {
      char *log = malloc(log_size);
      if (nvrtcGetProgramLog(prog, log) == NVRTC_SUCCESS) {
        fprintf(stderr,"Compilation log:\n%s\n", log);
      }
      free(log);
    }
    NVRTC_SUCCEED(res);
  }

  for (i = i_dyn; i < n_opts-num_extra_opts; i++) { free((char *)opts[i]); }
  free(opts);

  char *ptx;
  size_t ptx_size;
  NVRTC_SUCCEED(nvrtcGetPTXSize(prog, &ptx_size));
  ptx = malloc(ptx_size);
  NVRTC_SUCCEED(nvrtcGetPTX(prog, ptx));

  NVRTC_SUCCEED(nvrtcDestroyProgram(&prog));

  return ptx;
}

static void cuda_size_setup(struct cuda_context *ctx)
{
  if (ctx->cfg.default_block_size > ctx->max_block_size) {
    if (ctx->cfg.default_block_size_changed) {
      fprintf(stderr,
          "Note: Device limits default block size to %zu (down from %zu).\n",
          ctx->max_block_size, ctx->cfg.default_block_size);
    }
    ctx->cfg.default_block_size = ctx->max_block_size;
  }
  if (ctx->cfg.default_grid_size > ctx->max_grid_size) {
    if (ctx->cfg.default_grid_size_changed) {
      fprintf(stderr,
          "Note: Device limits default grid size to %zu (down from %zu).\n",
          ctx->max_grid_size, ctx->cfg.default_grid_size);
    }
    ctx->cfg.default_grid_size = ctx->max_grid_size;
  }
  if (ctx->cfg.default_tile_size > ctx->max_tile_size) {
    if (ctx->cfg.default_tile_size_changed) {
      fprintf(stderr,
          "Note: Device limits default tile size to %zu (down from %zu).\n",
          ctx->max_tile_size, ctx->cfg.default_tile_size);
    }
    ctx->cfg.default_tile_size = ctx->max_tile_size;
  }

  for (int i = 0; i < ctx->cfg.num_sizes; i++) {
    const char *size_class, *size_name;
    size_t *size_value, max_value, default_value;

    size_class = ctx->cfg.size_classes[i];
    size_value = &ctx->cfg.size_values[i];
    size_name = ctx->cfg.size_names[i];

    if (strstr(size_class, "group_size") == size_class) {
      max_value = ctx->max_block_size;
      default_value = ctx->cfg.default_block_size;
    } else if (strstr(size_class, "num_groups") == size_class) {
      max_value = ctx->max_grid_size;
      default_value = ctx->cfg.default_grid_size;
    } else if (strstr(size_class, "tile_size") == size_class) {
      max_value = ctx->max_tile_size;
      default_value = ctx->cfg.default_tile_size;
    } else if (strstr(size_class, "threshold") == size_class) {
      max_value = ctx->max_threshold;
      default_value = ctx->cfg.default_threshold;
    } else {
      panic(1, "Unknown size class for size '%s': %s\n", size_name, size_class);
    }

    if (*size_value == 0) {
      *size_value = default_value;
    } else if (max_value > 0 && *size_value > max_value) {
      fprintf(stderr, "Note: Device limits %s to %zu (down from %zu)\n",
              size_name, max_value, *size_value);
      *size_value = max_value;
    }
  }
}

static void dump_string_to_file(const char *file, const char *buf)
{
  FILE *f = fopen(file, "w");
  assert(f != NULL);
  assert(fputs(buf, f) != EOF);
  assert(fclose(f) == 0);
}

static void load_string_from_file(const char *file, char **obuf, size_t *olen)
{
  char *buf;
  size_t len;
  FILE *f = fopen(file, "r");

  assert(f != NULL);
  assert(fseek(f, 0, SEEK_END) == 0);
  len = ftell(f);
  assert(fseek(f, 0, SEEK_SET) == 0);

  buf = malloc(len + 1);
  assert(fread(buf, 1, len, f) == len);
  buf[len] = 0;
  *obuf = buf;
  if (olen != NULL) {
    *olen = len;
  }

  assert(fclose(f) == 0);
}

static void cuda_module_setup(struct cuda_context *ctx,
                              const char *src_fragments[],
                              const char *extra_opts[])
{
  char *ptx = NULL, *src = NULL;

  if (ctx->cfg.load_ptx_from == NULL && ctx->cfg.load_program_from == NULL) {
    src = concat_fragments(src_fragments);
    ptx = cuda_nvrtc_build(ctx, src, extra_opts);
  } else if (ctx->cfg.load_ptx_from == NULL) {
    load_string_from_file(ctx->cfg.load_program_from, &src, NULL);
    ptx = cuda_nvrtc_build(ctx, src, extra_opts);
  } else {
    if (ctx->cfg.load_program_from != NULL) {
      fprintf(stderr,
              "WARNING: Loading PTX from %s instead of C code from %s\n",
              ctx->cfg.load_ptx_from, ctx->cfg.load_program_from);
    }

    load_string_from_file(ctx->cfg.load_ptx_from, &ptx, NULL);
  }

  if (ctx->cfg.dump_program_to != NULL) {
    if (src == NULL) {
      src = concat_fragments(src_fragments);
    }
    dump_string_to_file(ctx->cfg.dump_program_to, src);
  }
  if (ctx->cfg.dump_ptx_to != NULL) {
    dump_string_to_file(ctx->cfg.dump_ptx_to, ptx);
  }

  CUDA_SUCCEED(cuModuleLoadData(&ctx->module, ptx));

  free(ptx);
  if (src != NULL) {
    free(src);
  }
}

void cuda_setup(struct cuda_context *ctx, const char *src_fragments[], const char *extra_opts[])
{
  CUDA_SUCCEED(cuInit(0));

  if (cuda_device_setup(ctx) != 0) {
    panic(-1, "No suitable CUDA device found.\n");
  }
  CUDA_SUCCEED(cuCtxCreate(&ctx->cu_ctx, 0, ctx->dev));

  free_list_init(&ctx->free_list);

  ctx->max_block_size = device_query(ctx->dev, MAX_THREADS_PER_BLOCK);
  ctx->max_grid_size = device_query(ctx->dev, MAX_GRID_DIM_X);
  ctx->max_tile_size = sqrt(ctx->max_block_size);
  ctx->max_threshold = 0;
  ctx->lockstep_width = device_query(ctx->dev, WARP_SIZE);

  cuda_size_setup(ctx);
  cuda_module_setup(ctx, src_fragments, extra_opts);
}

CUresult cuda_free_all(struct cuda_context *ctx);

void cuda_cleanup(struct cuda_context *ctx)
{
  CUDA_SUCCEED(cuda_free_all(ctx));
  CUDA_SUCCEED(cuModuleUnload(ctx->module));
  CUDA_SUCCEED(cuCtxDestroy(ctx->cu_ctx));
}

CUresult cuda_alloc(struct cuda_context *ctx, size_t min_size,
    const char *tag, CUdeviceptr *mem_out)
{
  if (min_size < sizeof(int)) {
    min_size = sizeof(int);
  }

  size_t size;
  if (free_list_find(&ctx->free_list, tag, &size, mem_out) == 0) {
    if (size >= min_size) {
      return CUDA_SUCCESS;
    } else {
      CUresult res = cuMemFree(*mem_out);
      if (res != CUDA_SUCCESS) {
        return res;
      }
    }
  }

  CUresult res = cuMemAlloc(mem_out, min_size);
  while (res == CUDA_ERROR_OUT_OF_MEMORY) {
    CUdeviceptr mem;
    if (free_list_first(&ctx->free_list, &mem) == 0) {
      res = cuMemFree(mem);
      if (res != CUDA_SUCCESS) {
        return res;
      }
    } else {
      break;
    }
    res = cuMemAlloc(mem_out, min_size);
  }

  return res;
}

CUresult cuda_free(struct cuda_context *ctx, CUdeviceptr mem,
    const char *tag)
{
  size_t size;
  CUdeviceptr existing_mem;

  // If there is already a block with this tag, then remove it.
  if (free_list_find(&ctx->free_list, tag, &size, &existing_mem) == 0) {
    CUresult res = cuMemFree(existing_mem);
    if (res != CUDA_SUCCESS) {
      return res;
    }
  }

  CUresult res = cuMemGetAddressRange(NULL, &size, mem);
  if (res == CUDA_SUCCESS) {
    free_list_insert(&ctx->free_list, size, mem, tag);
  }

  return res;
}

CUresult cuda_free_all(struct cuda_context *ctx) {
  CUdeviceptr mem;
  free_list_pack(&ctx->free_list);
  while (free_list_first(&ctx->free_list, &mem) == 0) {
    CUresult res = cuMemFree(mem);
    if (res != CUDA_SUCCESS) {
      return res;
    }
  }

  return CUDA_SUCCESS;
}


const char *cuda_program[] =
           {"typedef char int8_t;\ntypedef short int16_t;\ntypedef int int32_t;\ntypedef long int64_t;\ntypedef unsigned char uint8_t;\ntypedef unsigned short uint16_t;\ntypedef unsigned int uint32_t;\ntypedef unsigned long long uint64_t;\ntypedef uint8_t uchar;\ntypedef uint16_t ushort;\ntypedef uint32_t uint;\ntypedef uint64_t ulong;\n#define __kernel extern \"C\" __global__ __launch_bounds__(MAX_THREADS_PER_BLOCK)\n#define __global\n#define __local\n#define __private\n#define __constant\n#define __write_only\n#define __read_only\nstatic inline int get_group_id_fn(int block_dim0, int block_dim1,\n                                  int block_dim2, int d)\n{\n    switch (d) {\n        \n      case 0:\n        d = block_dim0;\n        break;\n        \n      case 1:\n        d = block_dim1;\n        break;\n        \n      case 2:\n        d = block_dim2;\n        break;\n    }\n    switch (d) {\n        \n      case 0:\n        return blockIdx.x;\n        \n      case 1:\n        return blockIdx.y;\n        \n      case 2:\n        return blockIdx.z;\n        \n      default:\n        return 0;\n    }\n}\n#define get_group_id(d) get_group_id_fn(block_dim0, block_dim1, block_dim2, d)\nstatic inline int get_num_groups_fn(int block_dim0, int block_dim1,\n                                    int block_dim2, int d)\n{\n    switch (d) {\n        \n      case 0:\n        d = block_dim0;\n        break;\n        \n      case 1:\n        d = block_dim1;\n        break;\n        \n      case 2:\n        d = block_dim2;\n        break;\n    }\n    switch (d) {\n        \n      case 0:\n        return gridDim.x;\n        \n      case 1:\n        return gridDim.y;\n        \n      case 2:\n        return gridDim.z;\n        \n      default:\n        return 0;\n    }\n}\n#define get_num_groups(d) get_num_groups_fn(block_dim0, block_dim1, block_dim2, d)\nstatic inline int get_local_id(int d)\n{\n    switch (d) {\n        \n      case 0:\n        return threadIdx.x;\n        \n      case 1:\n        return threadIdx.y;\n        \n      case 2:\n        return threadIdx.z;\n        \n      defau",
            "lt:\n        return 0;\n    }\n}\nstatic inline int get_local_size(int d)\n{\n    switch (d) {\n        \n      case 0:\n        return blockDim.x;\n        \n      case 1:\n        return blockDim.y;\n        \n      case 2:\n        return blockDim.z;\n        \n      default:\n        return 0;\n    }\n}\nstatic inline int get_global_id_fn(int block_dim0, int block_dim1,\n                                   int block_dim2, int d)\n{\n    return get_group_id(d) * get_local_size(d) + get_local_id(d);\n}\n#define get_global_id(d) get_global_id_fn(block_dim0, block_dim1, block_dim2, d)\nstatic inline int get_global_size(int block_dim0, int block_dim1,\n                                  int block_dim2, int d)\n{\n    return get_num_groups(d) * get_local_size(d);\n}\n#define CLK_LOCAL_MEM_FENCE 1\n#define CLK_GLOBAL_MEM_FENCE 2\nstatic inline void barrier(int x)\n{\n    __syncthreads();\n}\nstatic inline void mem_fence_local()\n{\n    __threadfence_block();\n}\nstatic inline void mem_fence_global()\n{\n    __threadfence();\n}\n#define NAN (0.0/0.0)\n#define INFINITY (1.0/0.0)\nextern volatile __shared__ char shared_mem[];\nstatic inline int atomic_add(volatile int *p, int val)\n{\n    return atomicAdd((int *) p, val);\n}\nstatic inline unsigned int atomic_add(volatile unsigned int *p,\n                                      unsigned int val)\n{\n    return atomicAdd((unsigned int *) p, val);\n}\nstatic inline unsigned long long atomic_add(volatile unsigned long long *p,\n                                            unsigned long long val)\n{\n    return atomicAdd((unsigned long long *) p, val);\n}\nstatic inline int atomic_max(volatile int *p, int val)\n{\n    return atomicMax((int *) p, val);\n}\nstatic inline unsigned int atomic_max(volatile unsigned int *p,\n                                      unsigned int val)\n{\n    return atomicMax((unsigned int *) p, val);\n}\nstatic inline unsigned long long atomic_max(volatile unsigned long long *p,\n                                            unsigned long long val)\n{\n    return atomicMax((unsigne",
            "d long long *) p, val);\n}\nstatic inline int atomic_min(volatile int *p, int val)\n{\n    return atomicMin((int *) p, val);\n}\nstatic inline unsigned int atomic_min(volatile unsigned int *p,\n                                      unsigned int val)\n{\n    return atomicMin((unsigned int *) p, val);\n}\nstatic inline unsigned long long atomic_min(volatile unsigned long long *p,\n                                            unsigned long long val)\n{\n    return atomicMin((unsigned long long *) p, val);\n}\nstatic inline int atomic_and(volatile int *p, int val)\n{\n    return atomicAnd((int *) p, val);\n}\nstatic inline unsigned int atomic_and(volatile unsigned int *p,\n                                      unsigned int val)\n{\n    return atomicAnd((unsigned int *) p, val);\n}\nstatic inline unsigned long long atomic_and(volatile unsigned long long *p,\n                                            unsigned long long val)\n{\n    return atomicAnd((unsigned long long *) p, val);\n}\nstatic inline int atomic_or(volatile int *p, int val)\n{\n    return atomicOr((int *) p, val);\n}\nstatic inline unsigned int atomic_or(volatile unsigned int *p, unsigned int val)\n{\n    return atomicOr((unsigned int *) p, val);\n}\nstatic inline unsigned long long atomic_or(volatile unsigned long long *p,\n                                           unsigned long long val)\n{\n    return atomicOr((unsigned long long *) p, val);\n}\nstatic inline int atomic_xor(volatile int *p, int val)\n{\n    return atomicXor((int *) p, val);\n}\nstatic inline unsigned int atomic_xor(volatile unsigned int *p,\n                                      unsigned int val)\n{\n    return atomicXor((unsigned int *) p, val);\n}\nstatic inline unsigned long long atomic_xor(volatile unsigned long long *p,\n                                            unsigned long long val)\n{\n    return atomicXor((unsigned long long *) p, val);\n}\nstatic inline int atomic_xchg(volatile int *p, int val)\n{\n    return atomicExch((int *) p, val);\n}\nstatic inline unsigned int atomic_xchg(volat",
            "ile unsigned int *p,\n                                       unsigned int val)\n{\n    return atomicExch((unsigned int *) p, val);\n}\nstatic inline unsigned long long atomic_xchg(volatile unsigned long long *p,\n                                             unsigned long long val)\n{\n    return atomicExch((unsigned long long *) p, val);\n}\nstatic inline int atomic_cmpxchg(volatile int *p, int cmp, int val)\n{\n    return atomicCAS((int *) p, cmp, val);\n}\nstatic inline unsigned int atomic_cmpxchg(volatile unsigned int *p,\n                                          unsigned int cmp, unsigned int val)\n{\n    return atomicCAS((unsigned int *) p, cmp, val);\n}\nstatic inline unsigned long long atomic_cmpxchg(volatile unsigned long long *p,\n                                                unsigned long long cmp,\n                                                unsigned long long val)\n{\n    return atomicCAS((unsigned long long *) p, cmp, val);\n}\nstatic inline int8_t add8(int8_t x, int8_t y)\n{\n    return x + y;\n}\nstatic inline int16_t add16(int16_t x, int16_t y)\n{\n    return x + y;\n}\nstatic inline int32_t add32(int32_t x, int32_t y)\n{\n    return x + y;\n}\nstatic inline int64_t add64(int64_t x, int64_t y)\n{\n    return x + y;\n}\nstatic inline int8_t sub8(int8_t x, int8_t y)\n{\n    return x - y;\n}\nstatic inline int16_t sub16(int16_t x, int16_t y)\n{\n    return x - y;\n}\nstatic inline int32_t sub32(int32_t x, int32_t y)\n{\n    return x - y;\n}\nstatic inline int64_t sub64(int64_t x, int64_t y)\n{\n    return x - y;\n}\nstatic inline int8_t mul8(int8_t x, int8_t y)\n{\n    return x * y;\n}\nstatic inline int16_t mul16(int16_t x, int16_t y)\n{\n    return x * y;\n}\nstatic inline int32_t mul32(int32_t x, int32_t y)\n{\n    return x * y;\n}\nstatic inline int64_t mul64(int64_t x, int64_t y)\n{\n    return x * y;\n}\nstatic inline uint8_t udiv8(uint8_t x, uint8_t y)\n{\n    return x / y;\n}\nstatic inline uint16_t udiv16(uint16_t x, uint16_t y)\n{\n    return x / y;\n}\nstatic inline uint32_t udiv32(uint32_t x, uint32_t y)\n{\n    ret",
            "urn x / y;\n}\nstatic inline uint64_t udiv64(uint64_t x, uint64_t y)\n{\n    return x / y;\n}\nstatic inline uint8_t umod8(uint8_t x, uint8_t y)\n{\n    return x % y;\n}\nstatic inline uint16_t umod16(uint16_t x, uint16_t y)\n{\n    return x % y;\n}\nstatic inline uint32_t umod32(uint32_t x, uint32_t y)\n{\n    return x % y;\n}\nstatic inline uint64_t umod64(uint64_t x, uint64_t y)\n{\n    return x % y;\n}\nstatic inline int8_t sdiv8(int8_t x, int8_t y)\n{\n    int8_t q = x / y;\n    int8_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int16_t sdiv16(int16_t x, int16_t y)\n{\n    int16_t q = x / y;\n    int16_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int32_t sdiv32(int32_t x, int32_t y)\n{\n    int32_t q = x / y;\n    int32_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int64_t sdiv64(int64_t x, int64_t y)\n{\n    int64_t q = x / y;\n    int64_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int8_t smod8(int8_t x, int8_t y)\n{\n    int8_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int16_t smod16(int16_t x, int16_t y)\n{\n    int16_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int32_t smod32(int32_t x, int32_t y)\n{\n    int32_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int64_t smod64(int64_t x, int64_t y)\n{\n    int64_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int8_t squot8(int8_t x, int8_t y)\n{\n    return x / y;\n}\nstatic inline int16_t squot16(int16_t x, int16_t y)\n{\n    return x / y;\n}\nstatic inline int32_t squot32(int32_t x, int32_t y)\n{\n    return x / y;\n}\nstatic inline int64_t squot64(int64_t x, int64_t y)\n{\n    return x / y;\n}\nstatic inline int8_t srem8(int8_t x, int8_t y)\n{\n    return x % y;\n}\ns",
            "tatic inline int16_t srem16(int16_t x, int16_t y)\n{\n    return x % y;\n}\nstatic inline int32_t srem32(int32_t x, int32_t y)\n{\n    return x % y;\n}\nstatic inline int64_t srem64(int64_t x, int64_t y)\n{\n    return x % y;\n}\nstatic inline int8_t smin8(int8_t x, int8_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int16_t smin16(int16_t x, int16_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int32_t smin32(int32_t x, int32_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int64_t smin64(int64_t x, int64_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint8_t umin8(uint8_t x, uint8_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint16_t umin16(uint16_t x, uint16_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint32_t umin32(uint32_t x, uint32_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint64_t umin64(uint64_t x, uint64_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int8_t smax8(int8_t x, int8_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int16_t smax16(int16_t x, int16_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int32_t smax32(int32_t x, int32_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int64_t smax64(int64_t x, int64_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint8_t umax8(uint8_t x, uint8_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint16_t umax16(uint16_t x, uint16_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint32_t umax32(uint32_t x, uint32_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint64_t umax64(uint64_t x, uint64_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint8_t shl8(uint8_t x, uint8_t y)\n{\n    return x << y;\n}\nstatic inline uint16_t shl16(uint16_t x, uint16_t y)\n{\n    return x << y;\n}\nstatic inline uint32_t shl32(uint32_t x, uint32_t y)\n{\n    return x << y;\n}\nstatic inline uint64_t shl64(uint64_t x, uint64_t y)\n{\n    return x << y;\n}\nstatic inline uint8_t lshr8(uint8_t x, uint8_t y)\n{\n    return x >> y;\n}\nstatic inline uint16_t lshr16(uint16_t x, uint16_t y)\n{\n    return x >> y;\n}\nstatic inline uint32_t lshr3",
            "2(uint32_t x, uint32_t y)\n{\n    return x >> y;\n}\nstatic inline uint64_t lshr64(uint64_t x, uint64_t y)\n{\n    return x >> y;\n}\nstatic inline int8_t ashr8(int8_t x, int8_t y)\n{\n    return x >> y;\n}\nstatic inline int16_t ashr16(int16_t x, int16_t y)\n{\n    return x >> y;\n}\nstatic inline int32_t ashr32(int32_t x, int32_t y)\n{\n    return x >> y;\n}\nstatic inline int64_t ashr64(int64_t x, int64_t y)\n{\n    return x >> y;\n}\nstatic inline uint8_t and8(uint8_t x, uint8_t y)\n{\n    return x & y;\n}\nstatic inline uint16_t and16(uint16_t x, uint16_t y)\n{\n    return x & y;\n}\nstatic inline uint32_t and32(uint32_t x, uint32_t y)\n{\n    return x & y;\n}\nstatic inline uint64_t and64(uint64_t x, uint64_t y)\n{\n    return x & y;\n}\nstatic inline uint8_t or8(uint8_t x, uint8_t y)\n{\n    return x | y;\n}\nstatic inline uint16_t or16(uint16_t x, uint16_t y)\n{\n    return x | y;\n}\nstatic inline uint32_t or32(uint32_t x, uint32_t y)\n{\n    return x | y;\n}\nstatic inline uint64_t or64(uint64_t x, uint64_t y)\n{\n    return x | y;\n}\nstatic inline uint8_t xor8(uint8_t x, uint8_t y)\n{\n    return x ^ y;\n}\nstatic inline uint16_t xor16(uint16_t x, uint16_t y)\n{\n    return x ^ y;\n}\nstatic inline uint32_t xor32(uint32_t x, uint32_t y)\n{\n    return x ^ y;\n}\nstatic inline uint64_t xor64(uint64_t x, uint64_t y)\n{\n    return x ^ y;\n}\nstatic inline char ult8(uint8_t x, uint8_t y)\n{\n    return x < y;\n}\nstatic inline char ult16(uint16_t x, uint16_t y)\n{\n    return x < y;\n}\nstatic inline char ult32(uint32_t x, uint32_t y)\n{\n    return x < y;\n}\nstatic inline char ult64(uint64_t x, uint64_t y)\n{\n    return x < y;\n}\nstatic inline char ule8(uint8_t x, uint8_t y)\n{\n    return x <= y;\n}\nstatic inline char ule16(uint16_t x, uint16_t y)\n{\n    return x <= y;\n}\nstatic inline char ule32(uint32_t x, uint32_t y)\n{\n    return x <= y;\n}\nstatic inline char ule64(uint64_t x, uint64_t y)\n{\n    return x <= y;\n}\nstatic inline char slt8(int8_t x, int8_t y)\n{\n    return x < y;\n}\nstatic inline char slt16(int16_t x, int16_t y)\n{\n    return x < y;",
            "\n}\nstatic inline char slt32(int32_t x, int32_t y)\n{\n    return x < y;\n}\nstatic inline char slt64(int64_t x, int64_t y)\n{\n    return x < y;\n}\nstatic inline char sle8(int8_t x, int8_t y)\n{\n    return x <= y;\n}\nstatic inline char sle16(int16_t x, int16_t y)\n{\n    return x <= y;\n}\nstatic inline char sle32(int32_t x, int32_t y)\n{\n    return x <= y;\n}\nstatic inline char sle64(int64_t x, int64_t y)\n{\n    return x <= y;\n}\nstatic inline int8_t pow8(int8_t x, int8_t y)\n{\n    int8_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int16_t pow16(int16_t x, int16_t y)\n{\n    int16_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int32_t pow32(int32_t x, int32_t y)\n{\n    int32_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int64_t pow64(int64_t x, int64_t y)\n{\n    int64_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline bool itob_i8_bool(int8_t x)\n{\n    return x;\n}\nstatic inline bool itob_i16_bool(int16_t x)\n{\n    return x;\n}\nstatic inline bool itob_i32_bool(int32_t x)\n{\n    return x;\n}\nstatic inline bool itob_i64_bool(int64_t x)\n{\n    return x;\n}\nstatic inline int8_t btoi_bool_i8(bool x)\n{\n    return x;\n}\nstatic inline int16_t btoi_bool_i16(bool x)\n{\n    return x;\n}\nstatic inline int32_t btoi_bool_i32(bool x)\n{\n    return x;\n}\nstatic inline int64_t btoi_bool_i64(bool x)\n{\n    return x;\n}\nstatic inline int8_t sext_i8_i8(int8_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i8_i16(int8_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i8_i32(int8_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i8_i64(int8_t x)\n{\n    return x;\n}\ns",
            "tatic inline int8_t sext_i16_i8(int16_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i16_i16(int16_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i16_i32(int16_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i16_i64(int16_t x)\n{\n    return x;\n}\nstatic inline int8_t sext_i32_i8(int32_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i32_i16(int32_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i32_i32(int32_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i32_i64(int32_t x)\n{\n    return x;\n}\nstatic inline int8_t sext_i64_i8(int64_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i64_i16(int64_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i64_i32(int64_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i64_i64(int64_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i8_i8(uint8_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i8_i16(uint8_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i8_i32(uint8_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i8_i64(uint8_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i16_i8(uint16_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i16_i16(uint16_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i16_i32(uint16_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i16_i64(uint16_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i32_i8(uint32_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i32_i16(uint32_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i32_i32(uint32_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i32_i64(uint32_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i64_i8(uint64_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i64_i16(uint64_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i64_i32(uint64_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i64_i64(uint64_t x)\n{\n    return x;\n}\nstatic inline float fdiv32(float x, float y)\n{\n    return x / y;\n}\nstatic inline float fadd32(float x, float y)\n{\n    return x + y;\n}\nstatic inline float fsub32(float x, float y)\n{\n    return x - y;\n",
            "}\nstatic inline float fmul32(float x, float y)\n{\n    return x * y;\n}\nstatic inline float fmin32(float x, float y)\n{\n    return x < y ? x : y;\n}\nstatic inline float fmax32(float x, float y)\n{\n    return x < y ? y : x;\n}\nstatic inline float fpow32(float x, float y)\n{\n    return pow(x, y);\n}\nstatic inline char cmplt32(float x, float y)\n{\n    return x < y;\n}\nstatic inline char cmple32(float x, float y)\n{\n    return x <= y;\n}\nstatic inline float sitofp_i8_f32(int8_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i16_f32(int16_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i32_f32(int32_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i64_f32(int64_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i8_f32(uint8_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i16_f32(uint16_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i32_f32(uint32_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i64_f32(uint64_t x)\n{\n    return x;\n}\nstatic inline int8_t fptosi_f32_i8(float x)\n{\n    return x;\n}\nstatic inline int16_t fptosi_f32_i16(float x)\n{\n    return x;\n}\nstatic inline int32_t fptosi_f32_i32(float x)\n{\n    return x;\n}\nstatic inline int64_t fptosi_f32_i64(float x)\n{\n    return x;\n}\nstatic inline uint8_t fptoui_f32_i8(float x)\n{\n    return x;\n}\nstatic inline uint16_t fptoui_f32_i16(float x)\n{\n    return x;\n}\nstatic inline uint32_t fptoui_f32_i32(float x)\n{\n    return x;\n}\nstatic inline uint64_t fptoui_f32_i64(float x)\n{\n    return x;\n}\nstatic inline float futrts_log32(float x)\n{\n    return log(x);\n}\nstatic inline float futrts_log2_32(float x)\n{\n    return log2(x);\n}\nstatic inline float futrts_log10_32(float x)\n{\n    return log10(x);\n}\nstatic inline float futrts_sqrt32(float x)\n{\n    return sqrt(x);\n}\nstatic inline float futrts_exp32(float x)\n{\n    return exp(x);\n}\nstatic inline float futrts_cos32(float x)\n{\n    return cos(x);\n}\nstatic inline float futrts_sin32(float x)\n{\n    return sin(x);\n}\nstatic inline float futrts_tan32(float x)\n{\n    return tan(x);\n}\nstatic inline float f",
            "utrts_acos32(float x)\n{\n    return acos(x);\n}\nstatic inline float futrts_asin32(float x)\n{\n    return asin(x);\n}\nstatic inline float futrts_atan32(float x)\n{\n    return atan(x);\n}\nstatic inline float futrts_atan2_32(float x, float y)\n{\n    return atan2(x, y);\n}\nstatic inline float futrts_round32(float x)\n{\n    return rint(x);\n}\nstatic inline char futrts_isnan32(float x)\n{\n    return isnan(x);\n}\nstatic inline char futrts_isinf32(float x)\n{\n    return isinf(x);\n}\nstatic inline int32_t futrts_to_bits32(float x)\n{\n    union {\n        float f;\n        int32_t t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\nstatic inline float futrts_from_bits32(int32_t x)\n{\n    union {\n        int32_t f;\n        float t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\nstatic inline double fdiv64(double x, double y)\n{\n    return x / y;\n}\nstatic inline double fadd64(double x, double y)\n{\n    return x + y;\n}\nstatic inline double fsub64(double x, double y)\n{\n    return x - y;\n}\nstatic inline double fmul64(double x, double y)\n{\n    return x * y;\n}\nstatic inline double fmin64(double x, double y)\n{\n    return x < y ? x : y;\n}\nstatic inline double fmax64(double x, double y)\n{\n    return x < y ? y : x;\n}\nstatic inline double fpow64(double x, double y)\n{\n    return pow(x, y);\n}\nstatic inline char cmplt64(double x, double y)\n{\n    return x < y;\n}\nstatic inline char cmple64(double x, double y)\n{\n    return x <= y;\n}\nstatic inline double sitofp_i8_f64(int8_t x)\n{\n    return x;\n}\nstatic inline double sitofp_i16_f64(int16_t x)\n{\n    return x;\n}\nstatic inline double sitofp_i32_f64(int32_t x)\n{\n    return x;\n}\nstatic inline double sitofp_i64_f64(int64_t x)\n{\n    return x;\n}\nstatic inline double uitofp_i8_f64(uint8_t x)\n{\n    return x;\n}\nstatic inline double uitofp_i16_f64(uint16_t x)\n{\n    return x;\n}\nstatic inline double uitofp_i32_f64(uint32_t x)\n{\n    return x;\n}\nstatic inline double uitofp_i64_f64(uint64_t x)\n{\n    return x;\n}\nstatic inline int8_t fptosi_f64_i8(double x)\n{\n    return x;\n}\nstatic inline ",
            "int16_t fptosi_f64_i16(double x)\n{\n    return x;\n}\nstatic inline int32_t fptosi_f64_i32(double x)\n{\n    return x;\n}\nstatic inline int64_t fptosi_f64_i64(double x)\n{\n    return x;\n}\nstatic inline uint8_t fptoui_f64_i8(double x)\n{\n    return x;\n}\nstatic inline uint16_t fptoui_f64_i16(double x)\n{\n    return x;\n}\nstatic inline uint32_t fptoui_f64_i32(double x)\n{\n    return x;\n}\nstatic inline uint64_t fptoui_f64_i64(double x)\n{\n    return x;\n}\nstatic inline double futrts_log64(double x)\n{\n    return log(x);\n}\nstatic inline double futrts_log2_64(double x)\n{\n    return log2(x);\n}\nstatic inline double futrts_log10_64(double x)\n{\n    return log10(x);\n}\nstatic inline double futrts_sqrt64(double x)\n{\n    return sqrt(x);\n}\nstatic inline double futrts_exp64(double x)\n{\n    return exp(x);\n}\nstatic inline double futrts_cos64(double x)\n{\n    return cos(x);\n}\nstatic inline double futrts_sin64(double x)\n{\n    return sin(x);\n}\nstatic inline double futrts_tan64(double x)\n{\n    return tan(x);\n}\nstatic inline double futrts_acos64(double x)\n{\n    return acos(x);\n}\nstatic inline double futrts_asin64(double x)\n{\n    return asin(x);\n}\nstatic inline double futrts_atan64(double x)\n{\n    return atan(x);\n}\nstatic inline double futrts_atan2_64(double x, double y)\n{\n    return atan2(x, y);\n}\nstatic inline double futrts_round64(double x)\n{\n    return rint(x);\n}\nstatic inline char futrts_isnan64(double x)\n{\n    return isnan(x);\n}\nstatic inline char futrts_isinf64(double x)\n{\n    return isinf(x);\n}\nstatic inline int64_t futrts_to_bits64(double x)\n{\n    union {\n        double f;\n        int64_t t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\nstatic inline double futrts_from_bits64(int64_t x)\n{\n    union {\n        int64_t f;\n        double t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\nstatic inline float fpconv_f32_f32(float x)\n{\n    return x;\n}\nstatic inline double fpconv_f32_f64(float x)\n{\n    return x;\n}\nstatic inline float fpconv_f64_f32(double x)\n{\n    return x;\n}\nstatic inline double fpconv_f64",
            "_f64(double x)\n{\n    return x;\n}\n__kernel void map_7803(int32_t o_7713, int32_t res_7714, int64_t res_7731,\n                       int32_t i_7759, __global unsigned char *bs_mem_7846,\n                       __global unsigned char *mem_7858, __global\n                       unsigned char *scs_mem_7864, __global\n                       unsigned char *mvs_mem_7866, __global\n                       unsigned char *pvs_mem_7868, __global\n                       unsigned char *mem_7871, __global\n                       unsigned char *mem_7874, __global\n                       unsigned char *mem_7877)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t wave_sizze_7921;\n    int32_t group_sizze_7922;\n    int32_t gtid_7796;\n    int32_t global_tid_7803;\n    int32_t local_tid_7804;\n    int32_t group_id_7805;\n    \n    global_tid_7803 = get_global_id(0);\n    local_tid_7804 = get_local_id(0);\n    group_sizze_7922 = get_local_size(0);\n    wave_sizze_7921 = LOCKSTEP_WIDTH;\n    group_id_7805 = get_group_id(0);\n    gtid_7796 = global_tid_7803;\n    \n    int64_t x_7807;\n    int64_t x_7808;\n    int32_t x_7809;\n    int32_t y_7810;\n    int32_t i_7811;\n    int16_t arg_7812;\n    int32_t res_7813;\n    int64_t res_7814;\n    int64_t res_7815;\n    int64_t x_7816;\n    int64_t x_7817;\n    int64_t x_7818;\n    int64_t res_7819;\n    int64_t complement_arg_7820;\n    int64_t y_7821;\n    int64_t res_7822;\n    int64_t res_7823;\n    int64_t arg_7824;\n    bool res_7825;\n    int32_t res_7826;\n    int64_t x_7832;\n    int64_t res_7833;\n    int64_t x_7834;\n    int64_t complement_arg_7835;\n    int64_t y_7836;\n    int64_t res_7837;\n    int64_t res_7838;\n    \n    if (slt32(gtid_7796, o_7713)) {\n        x_7807 = *(__global int64_t *) &mvs_mem_7866[gtid_7796 * 8];\n        x_7808 = *(__global int64_t *) &pvs_mem_7868[gtid_7796 * 8];\n        x_7809 = *(__global int32_t *) &scs_mem_7864[gtid_7796 * 4];\n        y_7810 = res_7714 * gtid_7796;\n        i_7811 = i_7759 + y_7810;",
            "\n        arg_7812 = *(__global int16_t *) &bs_mem_7846[i_7811 * 2];\n        res_7813 = zext_i16_i32(arg_7812);\n        res_7814 = *(__global int64_t *) &mem_7858[res_7813 * 8];\n        res_7815 = x_7807 | res_7814;\n        x_7816 = x_7808 & res_7814;\n        x_7817 = x_7808 + x_7816;\n        x_7818 = x_7808 ^ x_7817;\n        res_7819 = res_7814 | x_7818;\n        complement_arg_7820 = x_7808 | res_7819;\n        y_7821 = ~complement_arg_7820;\n        res_7822 = x_7807 | y_7821;\n        res_7823 = x_7808 & res_7819;\n        arg_7824 = res_7731 & res_7822;\n        res_7825 = itob_i64_bool(arg_7824);\n        if (res_7825) {\n            int32_t res_7827 = 1 + x_7809;\n            \n            res_7826 = res_7827;\n        } else {\n            int64_t arg_7828;\n            bool res_7829;\n            int32_t res_7830;\n            \n            arg_7828 = res_7731 & res_7823;\n            res_7829 = itob_i64_bool(arg_7828);\n            if (res_7829) {\n                int32_t res_7831 = x_7809 - 1;\n                \n                res_7830 = res_7831;\n            } else {\n                res_7830 = x_7809;\n            }\n            res_7826 = res_7830;\n        }\n        x_7832 = res_7822 << 1;\n        res_7833 = 1 | x_7832;\n        x_7834 = res_7823 << 1;\n        complement_arg_7835 = res_7815 | res_7833;\n        y_7836 = ~complement_arg_7835;\n        res_7837 = x_7834 | y_7836;\n        res_7838 = res_7815 & res_7833;\n    }\n    if (slt32(gtid_7796, o_7713)) {\n        *(__global int64_t *) &mem_7871[gtid_7796 * 8] = res_7838;\n    }\n    if (slt32(gtid_7796, o_7713)) {\n        *(__global int64_t *) &mem_7874[gtid_7796 * 8] = res_7837;\n    }\n    if (slt32(gtid_7796, o_7713)) {\n        *(__global int32_t *) &mem_7877[gtid_7796 * 4] = res_7826;\n    }\n}\n__kernel void replicate_7894(int32_t sizze_7709, int32_t o_7713, __global\n                             unsigned char *mem_7849)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t repli",
            "cate_gtid_7894;\n    int32_t replicate_ltid_7895;\n    int32_t replicate_gid_7896;\n    \n    replicate_gtid_7894 = get_global_id(0);\n    replicate_ltid_7895 = get_local_id(0);\n    replicate_gid_7896 = get_group_id(0);\n    if (slt32(replicate_gtid_7894, o_7713)) {\n        *(__global int32_t *) &mem_7849[replicate_gtid_7894 * 4] = sizze_7709;\n    }\n}\n__kernel void replicate_7899(int32_t o_7713, __global unsigned char *mem_7852)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t replicate_gtid_7899;\n    int32_t replicate_ltid_7900;\n    int32_t replicate_gid_7901;\n    \n    replicate_gtid_7899 = get_global_id(0);\n    replicate_ltid_7900 = get_local_id(0);\n    replicate_gid_7901 = get_group_id(0);\n    if (slt32(replicate_gtid_7899, o_7713)) {\n        *(__global int64_t *) &mem_7852[replicate_gtid_7899 * 8] = 0;\n    }\n}\n__kernel void replicate_7904(int32_t o_7713, __global unsigned char *mem_7855)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t replicate_gtid_7904;\n    int32_t replicate_ltid_7905;\n    int32_t replicate_gid_7906;\n    \n    replicate_gtid_7904 = get_global_id(0);\n    replicate_ltid_7905 = get_local_id(0);\n    replicate_gid_7906 = get_group_id(0);\n    if (slt32(replicate_gtid_7904, o_7713)) {\n        *(__global int64_t *) &mem_7855[replicate_gtid_7904 * 8] = -1;\n    }\n}\n__kernel void replicate_7909(__global unsigned char *mem_7858)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t replicate_gtid_7909;\n    int32_t replicate_ltid_7910;\n    int32_t replicate_gid_7911;\n    \n    replicate_gtid_7909 = get_global_id(0);\n    replicate_ltid_7910 = get_local_id(0);\n    replicate_gid_7911 = get_group_id(0);\n    if (slt32(replicate_gtid_7909, 256)) {\n        *(__global int64_t *) &mem_7858[replicate_gtid_7909 * 8] = 0;\n    }\n}\n",
            NULL};
struct memblock_device {
    int *references;
    CUdeviceptr mem;
    int64_t size;
    const char *desc;
} ;
struct memblock_local {
    int *references;
    unsigned char mem;
    int64_t size;
    const char *desc;
} ;
struct memblock {
    int *references;
    char *mem;
    int64_t size;
    const char *desc;
} ;
static const char *size_names[] = {"main.group_size_7797",
                                   "main.group_size_7897",
                                   "main.group_size_7902",
                                   "main.group_size_7907",
                                   "main.group_size_7912"};
static const char *size_vars[] = {"mainzigroup_sizze_7797",
                                  "mainzigroup_sizze_7897",
                                  "mainzigroup_sizze_7902",
                                  "mainzigroup_sizze_7907",
                                  "mainzigroup_sizze_7912"};
static const char *size_classes[] = {"group_size", "group_size", "group_size",
                                     "group_size", "group_size"};
int futhark_get_num_sizes(void)
{
    return 5;
}
const char *futhark_get_size_name(int i)
{
    return size_names[i];
}
const char *futhark_get_size_class(int i)
{
    return size_classes[i];
}
struct sizes {
    size_t mainzigroup_sizze_7797;
    size_t mainzigroup_sizze_7897;
    size_t mainzigroup_sizze_7902;
    size_t mainzigroup_sizze_7907;
    size_t mainzigroup_sizze_7912;
} ;
struct futhark_context_config {
    struct cuda_config cu_cfg;
    size_t sizes[5];
    int num_nvrtc_opts;
    const char **nvrtc_opts;
} ;
struct futhark_context_config *futhark_context_config_new(void)
{
    struct futhark_context_config *cfg =
                                  malloc(sizeof(struct futhark_context_config));
    
    if (cfg == NULL)
        return NULL;
    cfg->num_nvrtc_opts = 0;
    cfg->nvrtc_opts = malloc(sizeof(const char *));
    cfg->nvrtc_opts[0] = NULL;
    cfg->sizes[0] = 0;
    cfg->sizes[1] = 0;
    cfg->sizes[2] = 0;
    cfg->sizes[3] = 0;
    cfg->sizes[4] = 0;
    cuda_config_init(&cfg->cu_cfg, 5, size_names, size_vars, cfg->sizes,
                     size_classes);
    return cfg;
}
void futhark_context_config_free(struct futhark_context_config *cfg)
{
    free(cfg->nvrtc_opts);
    free(cfg);
}
void futhark_context_config_add_nvrtc_option(struct futhark_context_config *cfg,
                                             const char *opt)
{
    cfg->nvrtc_opts[cfg->num_nvrtc_opts] = opt;
    cfg->num_nvrtc_opts++;
    cfg->nvrtc_opts = realloc(cfg->nvrtc_opts, (cfg->num_nvrtc_opts + 1) *
                              sizeof(const char *));
    cfg->nvrtc_opts[cfg->num_nvrtc_opts] = NULL;
}
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag)
{
    cfg->cu_cfg.logging = cfg->cu_cfg.debugging = flag;
}
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag)
{
    cfg->cu_cfg.logging = flag;
}
void futhark_context_config_set_device(struct futhark_context_config *cfg, const
                                       char *s)
{
    set_preferred_device(&cfg->cu_cfg, s);
}
void futhark_context_config_dump_program_to(struct futhark_context_config *cfg,
                                            const char *path)
{
    cfg->cu_cfg.dump_program_to = path;
}
void futhark_context_config_load_program_from(struct futhark_context_config *cfg,
                                              const char *path)
{
    cfg->cu_cfg.load_program_from = path;
}
void futhark_context_config_dump_ptx_to(struct futhark_context_config *cfg,
                                        const char *path)
{
    cfg->cu_cfg.dump_ptx_to = path;
}
void futhark_context_config_load_ptx_from(struct futhark_context_config *cfg,
                                          const char *path)
{
    cfg->cu_cfg.load_ptx_from = path;
}
void futhark_context_config_set_default_block_size(struct futhark_context_config *cfg,
                                                   int size)
{
    cfg->cu_cfg.default_block_size = size;
    cfg->cu_cfg.default_block_size_changed = 1;
}
void futhark_context_config_set_default_grid_size(struct futhark_context_config *cfg,
                                                  int num)
{
    cfg->cu_cfg.default_grid_size = num;
}
void futhark_context_config_set_default_tile_size(struct futhark_context_config *cfg,
                                                  int size)
{
    cfg->cu_cfg.default_tile_size = size;
    cfg->cu_cfg.default_tile_size_changed = 1;
}
void futhark_context_config_set_default_threshold(struct futhark_context_config *cfg,
                                                  int size)
{
    cfg->cu_cfg.default_threshold = size;
}
int futhark_context_config_set_size(struct futhark_context_config *cfg, const
                                    char *size_name, size_t size_value)
{
    for (int i = 0; i < 5; i++) {
        if (strcmp(size_name, size_names[i]) == 0) {
            cfg->sizes[i] = size_value;
            return 0;
        }
    }
    return 1;
}
struct futhark_context {
    int detail_memory;
    int debugging;
    lock_t lock;
    char *error;
    int64_t peak_mem_usage_device;
    int64_t cur_mem_usage_device;
    int64_t peak_mem_usage_local;
    int64_t cur_mem_usage_local;
    int64_t peak_mem_usage_default;
    int64_t cur_mem_usage_default;
    CUfunction map_7803;
    CUfunction replicate_7894;
    CUfunction replicate_7899;
    CUfunction replicate_7904;
    CUfunction replicate_7909;
    struct cuda_context cuda;
    struct sizes sizes;
} ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg)
{
    struct futhark_context *ctx = malloc(sizeof(struct futhark_context));
    
    if (ctx == NULL)
        return NULL;
    ctx->debugging = ctx->detail_memory = cfg->cu_cfg.debugging;
    ctx->cuda.cfg = cfg->cu_cfg;
    create_lock(&ctx->lock);
    ctx->peak_mem_usage_device = 0;
    ctx->cur_mem_usage_device = 0;
    ctx->peak_mem_usage_local = 0;
    ctx->cur_mem_usage_local = 0;
    ctx->peak_mem_usage_default = 0;
    ctx->cur_mem_usage_default = 0;
    cuda_setup(&ctx->cuda, cuda_program, cfg->nvrtc_opts);
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->map_7803, ctx->cuda.module,
                                     "map_7803"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->replicate_7894, ctx->cuda.module,
                                     "replicate_7894"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->replicate_7899, ctx->cuda.module,
                                     "replicate_7899"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->replicate_7904, ctx->cuda.module,
                                     "replicate_7904"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->replicate_7909, ctx->cuda.module,
                                     "replicate_7909"));
    ctx->sizes.mainzigroup_sizze_7797 = cfg->sizes[0];
    ctx->sizes.mainzigroup_sizze_7897 = cfg->sizes[1];
    ctx->sizes.mainzigroup_sizze_7902 = cfg->sizes[2];
    ctx->sizes.mainzigroup_sizze_7907 = cfg->sizes[3];
    ctx->sizes.mainzigroup_sizze_7912 = cfg->sizes[4];
    return ctx;
}
void futhark_context_free(struct futhark_context *ctx)
{
    cuda_cleanup(&ctx->cuda);
    free_lock(&ctx->lock);
    free(ctx);
}
int futhark_context_sync(struct futhark_context *ctx)
{
    CUDA_SUCCEED(cuCtxSynchronize());
    return 0;
}
char *futhark_context_get_error(struct futhark_context *ctx)
{
    return ctx->error;
}
static int memblock_unref_device(struct futhark_context *ctx,
                                 struct memblock_device *block, const
                                 char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(stderr,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "space 'device'", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_device -= block->size;
            CUDA_SUCCEED(cuda_free(&ctx->cuda, block->mem, block->desc));
            free(block->references);
            if (ctx->detail_memory)
                fprintf(stderr,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_device);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc_device(struct futhark_context *ctx,
                                 struct memblock_device *block, int64_t size,
                                 const char *desc)
{
    if (size < 0)
        panic(1, "Negative allocation of %lld bytes attempted for %s in %s.\n",
              (long long) size, desc, "space 'device'",
              ctx->cur_mem_usage_device);
    
    int ret = memblock_unref_device(ctx, block, desc);
    
    CUDA_SUCCEED(cuda_alloc(&ctx->cuda, size, desc, &block->mem));
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    ctx->cur_mem_usage_device += size;
    if (ctx->detail_memory)
        fprintf(stderr,
                "Allocated %lld bytes for %s in %s (now allocated: %lld bytes)",
                (long long) size, desc, "space 'device'",
                (long long) ctx->cur_mem_usage_device);
    if (ctx->cur_mem_usage_device > ctx->peak_mem_usage_device) {
        ctx->peak_mem_usage_device = ctx->cur_mem_usage_device;
        if (ctx->detail_memory)
            fprintf(stderr, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(stderr, ".\n");
    return ret;
}
static int memblock_set_device(struct futhark_context *ctx,
                               struct memblock_device *lhs,
                               struct memblock_device *rhs, const
                               char *lhs_desc)
{
    int ret = memblock_unref_device(ctx, lhs, lhs_desc);
    
    (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
static int memblock_unref_local(struct futhark_context *ctx,
                                struct memblock_local *block, const char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(stderr,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "space 'local'", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_local -= block->size;
            free(block->references);
            if (ctx->detail_memory)
                fprintf(stderr,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_local);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc_local(struct futhark_context *ctx,
                                struct memblock_local *block, int64_t size,
                                const char *desc)
{
    if (size < 0)
        panic(1, "Negative allocation of %lld bytes attempted for %s in %s.\n",
              (long long) size, desc, "space 'local'",
              ctx->cur_mem_usage_local);
    
    int ret = memblock_unref_local(ctx, block, desc);
    
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    ctx->cur_mem_usage_local += size;
    if (ctx->detail_memory)
        fprintf(stderr,
                "Allocated %lld bytes for %s in %s (now allocated: %lld bytes)",
                (long long) size, desc, "space 'local'",
                (long long) ctx->cur_mem_usage_local);
    if (ctx->cur_mem_usage_local > ctx->peak_mem_usage_local) {
        ctx->peak_mem_usage_local = ctx->cur_mem_usage_local;
        if (ctx->detail_memory)
            fprintf(stderr, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(stderr, ".\n");
    return ret;
}
static int memblock_set_local(struct futhark_context *ctx,
                              struct memblock_local *lhs,
                              struct memblock_local *rhs, const char *lhs_desc)
{
    int ret = memblock_unref_local(ctx, lhs, lhs_desc);
    
    (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
static int memblock_unref(struct futhark_context *ctx, struct memblock *block,
                          const char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(stderr,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "default space", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_default -= block->size;
            free(block->mem);
            free(block->references);
            if (ctx->detail_memory)
                fprintf(stderr,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_default);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc(struct futhark_context *ctx, struct memblock *block,
                          int64_t size, const char *desc)
{
    if (size < 0)
        panic(1, "Negative allocation of %lld bytes attempted for %s in %s.\n",
              (long long) size, desc, "default space",
              ctx->cur_mem_usage_default);
    
    int ret = memblock_unref(ctx, block, desc);
    
    block->mem = (char *) malloc(size);
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    ctx->cur_mem_usage_default += size;
    if (ctx->detail_memory)
        fprintf(stderr,
                "Allocated %lld bytes for %s in %s (now allocated: %lld bytes)",
                (long long) size, desc, "default space",
                (long long) ctx->cur_mem_usage_default);
    if (ctx->cur_mem_usage_default > ctx->peak_mem_usage_default) {
        ctx->peak_mem_usage_default = ctx->cur_mem_usage_default;
        if (ctx->detail_memory)
            fprintf(stderr, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(stderr, ".\n");
    return ret;
}
static int memblock_set(struct futhark_context *ctx, struct memblock *lhs,
                        struct memblock *rhs, const char *lhs_desc)
{
    int ret = memblock_unref(ctx, lhs, lhs_desc);
    
    (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
void futhark_debugging_report(struct futhark_context *ctx)
{
    if (ctx->detail_memory) {
        fprintf(stderr, "Peak memory usage for space 'device': %lld bytes.\n",
                (long long) ctx->peak_mem_usage_device);
        fprintf(stderr, "Peak memory usage for space 'local': %lld bytes.\n",
                (long long) ctx->peak_mem_usage_local);
        fprintf(stderr, "Peak memory usage for default space: %lld bytes.\n",
                (long long) ctx->peak_mem_usage_default);
    }
    if (ctx->debugging) { }
}
static int futrts_main(struct futhark_context *ctx,
                       int64_t *out_out_memsizze_7923,
                       struct memblock_device *out_mem_p_7924,
                       int32_t *out_out_arrsizze_7925,
                       int64_t *out_out_memsizze_7926,
                       struct memblock_device *out_mem_p_7927,
                       int32_t *out_out_arrsizze_7928,
                       int64_t *out_out_memsizze_7929,
                       struct memblock_device *out_mem_p_7930,
                       int32_t *out_out_arrsizze_7931, int64_t a_mem_sizze_7843,
                       struct memblock_device a_mem_7844,
                       int64_t bs_mem_sizze_7845,
                       struct memblock_device bs_mem_7846, int32_t sizze_7709,
                       int32_t sizze_7710, int32_t o_7713);
static inline int8_t add8(int8_t x, int8_t y)
{
    return x + y;
}
static inline int16_t add16(int16_t x, int16_t y)
{
    return x + y;
}
static inline int32_t add32(int32_t x, int32_t y)
{
    return x + y;
}
static inline int64_t add64(int64_t x, int64_t y)
{
    return x + y;
}
static inline int8_t sub8(int8_t x, int8_t y)
{
    return x - y;
}
static inline int16_t sub16(int16_t x, int16_t y)
{
    return x - y;
}
static inline int32_t sub32(int32_t x, int32_t y)
{
    return x - y;
}
static inline int64_t sub64(int64_t x, int64_t y)
{
    return x - y;
}
static inline int8_t mul8(int8_t x, int8_t y)
{
    return x * y;
}
static inline int16_t mul16(int16_t x, int16_t y)
{
    return x * y;
}
static inline int32_t mul32(int32_t x, int32_t y)
{
    return x * y;
}
static inline int64_t mul64(int64_t x, int64_t y)
{
    return x * y;
}
static inline uint8_t udiv8(uint8_t x, uint8_t y)
{
    return x / y;
}
static inline uint16_t udiv16(uint16_t x, uint16_t y)
{
    return x / y;
}
static inline uint32_t udiv32(uint32_t x, uint32_t y)
{
    return x / y;
}
static inline uint64_t udiv64(uint64_t x, uint64_t y)
{
    return x / y;
}
static inline uint8_t umod8(uint8_t x, uint8_t y)
{
    return x % y;
}
static inline uint16_t umod16(uint16_t x, uint16_t y)
{
    return x % y;
}
static inline uint32_t umod32(uint32_t x, uint32_t y)
{
    return x % y;
}
static inline uint64_t umod64(uint64_t x, uint64_t y)
{
    return x % y;
}
static inline int8_t sdiv8(int8_t x, int8_t y)
{
    int8_t q = x / y;
    int8_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int16_t sdiv16(int16_t x, int16_t y)
{
    int16_t q = x / y;
    int16_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int32_t sdiv32(int32_t x, int32_t y)
{
    int32_t q = x / y;
    int32_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int64_t sdiv64(int64_t x, int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int8_t smod8(int8_t x, int8_t y)
{
    int8_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int16_t smod16(int16_t x, int16_t y)
{
    int16_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int32_t smod32(int32_t x, int32_t y)
{
    int32_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int64_t smod64(int64_t x, int64_t y)
{
    int64_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int8_t squot8(int8_t x, int8_t y)
{
    return x / y;
}
static inline int16_t squot16(int16_t x, int16_t y)
{
    return x / y;
}
static inline int32_t squot32(int32_t x, int32_t y)
{
    return x / y;
}
static inline int64_t squot64(int64_t x, int64_t y)
{
    return x / y;
}
static inline int8_t srem8(int8_t x, int8_t y)
{
    return x % y;
}
static inline int16_t srem16(int16_t x, int16_t y)
{
    return x % y;
}
static inline int32_t srem32(int32_t x, int32_t y)
{
    return x % y;
}
static inline int64_t srem64(int64_t x, int64_t y)
{
    return x % y;
}
static inline int8_t smin8(int8_t x, int8_t y)
{
    return x < y ? x : y;
}
static inline int16_t smin16(int16_t x, int16_t y)
{
    return x < y ? x : y;
}
static inline int32_t smin32(int32_t x, int32_t y)
{
    return x < y ? x : y;
}
static inline int64_t smin64(int64_t x, int64_t y)
{
    return x < y ? x : y;
}
static inline uint8_t umin8(uint8_t x, uint8_t y)
{
    return x < y ? x : y;
}
static inline uint16_t umin16(uint16_t x, uint16_t y)
{
    return x < y ? x : y;
}
static inline uint32_t umin32(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}
static inline uint64_t umin64(uint64_t x, uint64_t y)
{
    return x < y ? x : y;
}
static inline int8_t smax8(int8_t x, int8_t y)
{
    return x < y ? y : x;
}
static inline int16_t smax16(int16_t x, int16_t y)
{
    return x < y ? y : x;
}
static inline int32_t smax32(int32_t x, int32_t y)
{
    return x < y ? y : x;
}
static inline int64_t smax64(int64_t x, int64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t umax8(uint8_t x, uint8_t y)
{
    return x < y ? y : x;
}
static inline uint16_t umax16(uint16_t x, uint16_t y)
{
    return x < y ? y : x;
}
static inline uint32_t umax32(uint32_t x, uint32_t y)
{
    return x < y ? y : x;
}
static inline uint64_t umax64(uint64_t x, uint64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t shl8(uint8_t x, uint8_t y)
{
    return x << y;
}
static inline uint16_t shl16(uint16_t x, uint16_t y)
{
    return x << y;
}
static inline uint32_t shl32(uint32_t x, uint32_t y)
{
    return x << y;
}
static inline uint64_t shl64(uint64_t x, uint64_t y)
{
    return x << y;
}
static inline uint8_t lshr8(uint8_t x, uint8_t y)
{
    return x >> y;
}
static inline uint16_t lshr16(uint16_t x, uint16_t y)
{
    return x >> y;
}
static inline uint32_t lshr32(uint32_t x, uint32_t y)
{
    return x >> y;
}
static inline uint64_t lshr64(uint64_t x, uint64_t y)
{
    return x >> y;
}
static inline int8_t ashr8(int8_t x, int8_t y)
{
    return x >> y;
}
static inline int16_t ashr16(int16_t x, int16_t y)
{
    return x >> y;
}
static inline int32_t ashr32(int32_t x, int32_t y)
{
    return x >> y;
}
static inline int64_t ashr64(int64_t x, int64_t y)
{
    return x >> y;
}
static inline uint8_t and8(uint8_t x, uint8_t y)
{
    return x & y;
}
static inline uint16_t and16(uint16_t x, uint16_t y)
{
    return x & y;
}
static inline uint32_t and32(uint32_t x, uint32_t y)
{
    return x & y;
}
static inline uint64_t and64(uint64_t x, uint64_t y)
{
    return x & y;
}
static inline uint8_t or8(uint8_t x, uint8_t y)
{
    return x | y;
}
static inline uint16_t or16(uint16_t x, uint16_t y)
{
    return x | y;
}
static inline uint32_t or32(uint32_t x, uint32_t y)
{
    return x | y;
}
static inline uint64_t or64(uint64_t x, uint64_t y)
{
    return x | y;
}
static inline uint8_t xor8(uint8_t x, uint8_t y)
{
    return x ^ y;
}
static inline uint16_t xor16(uint16_t x, uint16_t y)
{
    return x ^ y;
}
static inline uint32_t xor32(uint32_t x, uint32_t y)
{
    return x ^ y;
}
static inline uint64_t xor64(uint64_t x, uint64_t y)
{
    return x ^ y;
}
static inline char ult8(uint8_t x, uint8_t y)
{
    return x < y;
}
static inline char ult16(uint16_t x, uint16_t y)
{
    return x < y;
}
static inline char ult32(uint32_t x, uint32_t y)
{
    return x < y;
}
static inline char ult64(uint64_t x, uint64_t y)
{
    return x < y;
}
static inline char ule8(uint8_t x, uint8_t y)
{
    return x <= y;
}
static inline char ule16(uint16_t x, uint16_t y)
{
    return x <= y;
}
static inline char ule32(uint32_t x, uint32_t y)
{
    return x <= y;
}
static inline char ule64(uint64_t x, uint64_t y)
{
    return x <= y;
}
static inline char slt8(int8_t x, int8_t y)
{
    return x < y;
}
static inline char slt16(int16_t x, int16_t y)
{
    return x < y;
}
static inline char slt32(int32_t x, int32_t y)
{
    return x < y;
}
static inline char slt64(int64_t x, int64_t y)
{
    return x < y;
}
static inline char sle8(int8_t x, int8_t y)
{
    return x <= y;
}
static inline char sle16(int16_t x, int16_t y)
{
    return x <= y;
}
static inline char sle32(int32_t x, int32_t y)
{
    return x <= y;
}
static inline char sle64(int64_t x, int64_t y)
{
    return x <= y;
}
static inline int8_t pow8(int8_t x, int8_t y)
{
    int8_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int16_t pow16(int16_t x, int16_t y)
{
    int16_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int32_t pow32(int32_t x, int32_t y)
{
    int32_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int64_t pow64(int64_t x, int64_t y)
{
    int64_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline bool itob_i8_bool(int8_t x)
{
    return x;
}
static inline bool itob_i16_bool(int16_t x)
{
    return x;
}
static inline bool itob_i32_bool(int32_t x)
{
    return x;
}
static inline bool itob_i64_bool(int64_t x)
{
    return x;
}
static inline int8_t btoi_bool_i8(bool x)
{
    return x;
}
static inline int16_t btoi_bool_i16(bool x)
{
    return x;
}
static inline int32_t btoi_bool_i32(bool x)
{
    return x;
}
static inline int64_t btoi_bool_i64(bool x)
{
    return x;
}
static inline int8_t sext_i8_i8(int8_t x)
{
    return x;
}
static inline int16_t sext_i8_i16(int8_t x)
{
    return x;
}
static inline int32_t sext_i8_i32(int8_t x)
{
    return x;
}
static inline int64_t sext_i8_i64(int8_t x)
{
    return x;
}
static inline int8_t sext_i16_i8(int16_t x)
{
    return x;
}
static inline int16_t sext_i16_i16(int16_t x)
{
    return x;
}
static inline int32_t sext_i16_i32(int16_t x)
{
    return x;
}
static inline int64_t sext_i16_i64(int16_t x)
{
    return x;
}
static inline int8_t sext_i32_i8(int32_t x)
{
    return x;
}
static inline int16_t sext_i32_i16(int32_t x)
{
    return x;
}
static inline int32_t sext_i32_i32(int32_t x)
{
    return x;
}
static inline int64_t sext_i32_i64(int32_t x)
{
    return x;
}
static inline int8_t sext_i64_i8(int64_t x)
{
    return x;
}
static inline int16_t sext_i64_i16(int64_t x)
{
    return x;
}
static inline int32_t sext_i64_i32(int64_t x)
{
    return x;
}
static inline int64_t sext_i64_i64(int64_t x)
{
    return x;
}
static inline uint8_t zext_i8_i8(uint8_t x)
{
    return x;
}
static inline uint16_t zext_i8_i16(uint8_t x)
{
    return x;
}
static inline uint32_t zext_i8_i32(uint8_t x)
{
    return x;
}
static inline uint64_t zext_i8_i64(uint8_t x)
{
    return x;
}
static inline uint8_t zext_i16_i8(uint16_t x)
{
    return x;
}
static inline uint16_t zext_i16_i16(uint16_t x)
{
    return x;
}
static inline uint32_t zext_i16_i32(uint16_t x)
{
    return x;
}
static inline uint64_t zext_i16_i64(uint16_t x)
{
    return x;
}
static inline uint8_t zext_i32_i8(uint32_t x)
{
    return x;
}
static inline uint16_t zext_i32_i16(uint32_t x)
{
    return x;
}
static inline uint32_t zext_i32_i32(uint32_t x)
{
    return x;
}
static inline uint64_t zext_i32_i64(uint32_t x)
{
    return x;
}
static inline uint8_t zext_i64_i8(uint64_t x)
{
    return x;
}
static inline uint16_t zext_i64_i16(uint64_t x)
{
    return x;
}
static inline uint32_t zext_i64_i32(uint64_t x)
{
    return x;
}
static inline uint64_t zext_i64_i64(uint64_t x)
{
    return x;
}
static inline float fdiv32(float x, float y)
{
    return x / y;
}
static inline float fadd32(float x, float y)
{
    return x + y;
}
static inline float fsub32(float x, float y)
{
    return x - y;
}
static inline float fmul32(float x, float y)
{
    return x * y;
}
static inline float fmin32(float x, float y)
{
    return x < y ? x : y;
}
static inline float fmax32(float x, float y)
{
    return x < y ? y : x;
}
static inline float fpow32(float x, float y)
{
    return pow(x, y);
}
static inline char cmplt32(float x, float y)
{
    return x < y;
}
static inline char cmple32(float x, float y)
{
    return x <= y;
}
static inline float sitofp_i8_f32(int8_t x)
{
    return x;
}
static inline float sitofp_i16_f32(int16_t x)
{
    return x;
}
static inline float sitofp_i32_f32(int32_t x)
{
    return x;
}
static inline float sitofp_i64_f32(int64_t x)
{
    return x;
}
static inline float uitofp_i8_f32(uint8_t x)
{
    return x;
}
static inline float uitofp_i16_f32(uint16_t x)
{
    return x;
}
static inline float uitofp_i32_f32(uint32_t x)
{
    return x;
}
static inline float uitofp_i64_f32(uint64_t x)
{
    return x;
}
static inline int8_t fptosi_f32_i8(float x)
{
    return x;
}
static inline int16_t fptosi_f32_i16(float x)
{
    return x;
}
static inline int32_t fptosi_f32_i32(float x)
{
    return x;
}
static inline int64_t fptosi_f32_i64(float x)
{
    return x;
}
static inline uint8_t fptoui_f32_i8(float x)
{
    return x;
}
static inline uint16_t fptoui_f32_i16(float x)
{
    return x;
}
static inline uint32_t fptoui_f32_i32(float x)
{
    return x;
}
static inline uint64_t fptoui_f32_i64(float x)
{
    return x;
}
static inline double fdiv64(double x, double y)
{
    return x / y;
}
static inline double fadd64(double x, double y)
{
    return x + y;
}
static inline double fsub64(double x, double y)
{
    return x - y;
}
static inline double fmul64(double x, double y)
{
    return x * y;
}
static inline double fmin64(double x, double y)
{
    return x < y ? x : y;
}
static inline double fmax64(double x, double y)
{
    return x < y ? y : x;
}
static inline double fpow64(double x, double y)
{
    return pow(x, y);
}
static inline char cmplt64(double x, double y)
{
    return x < y;
}
static inline char cmple64(double x, double y)
{
    return x <= y;
}
static inline double sitofp_i8_f64(int8_t x)
{
    return x;
}
static inline double sitofp_i16_f64(int16_t x)
{
    return x;
}
static inline double sitofp_i32_f64(int32_t x)
{
    return x;
}
static inline double sitofp_i64_f64(int64_t x)
{
    return x;
}
static inline double uitofp_i8_f64(uint8_t x)
{
    return x;
}
static inline double uitofp_i16_f64(uint16_t x)
{
    return x;
}
static inline double uitofp_i32_f64(uint32_t x)
{
    return x;
}
static inline double uitofp_i64_f64(uint64_t x)
{
    return x;
}
static inline int8_t fptosi_f64_i8(double x)
{
    return x;
}
static inline int16_t fptosi_f64_i16(double x)
{
    return x;
}
static inline int32_t fptosi_f64_i32(double x)
{
    return x;
}
static inline int64_t fptosi_f64_i64(double x)
{
    return x;
}
static inline uint8_t fptoui_f64_i8(double x)
{
    return x;
}
static inline uint16_t fptoui_f64_i16(double x)
{
    return x;
}
static inline uint32_t fptoui_f64_i32(double x)
{
    return x;
}
static inline uint64_t fptoui_f64_i64(double x)
{
    return x;
}
static inline float fpconv_f32_f32(float x)
{
    return x;
}
static inline double fpconv_f32_f64(float x)
{
    return x;
}
static inline float fpconv_f64_f32(double x)
{
    return x;
}
static inline double fpconv_f64_f64(double x)
{
    return x;
}
static inline float futrts_log32(float x)
{
    return log(x);
}
static inline float futrts_log2_32(float x)
{
    return log2(x);
}
static inline float futrts_log10_32(float x)
{
    return log10(x);
}
static inline float futrts_sqrt32(float x)
{
    return sqrt(x);
}
static inline float futrts_exp32(float x)
{
    return exp(x);
}
static inline float futrts_cos32(float x)
{
    return cos(x);
}
static inline float futrts_sin32(float x)
{
    return sin(x);
}
static inline float futrts_tan32(float x)
{
    return tan(x);
}
static inline float futrts_acos32(float x)
{
    return acos(x);
}
static inline float futrts_asin32(float x)
{
    return asin(x);
}
static inline float futrts_atan32(float x)
{
    return atan(x);
}
static inline float futrts_atan2_32(float x, float y)
{
    return atan2(x, y);
}
static inline float futrts_round32(float x)
{
    return rint(x);
}
static inline char futrts_isnan32(float x)
{
    return isnan(x);
}
static inline char futrts_isinf32(float x)
{
    return isinf(x);
}
static inline int32_t futrts_to_bits32(float x)
{
    union {
        float f;
        int32_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline float futrts_from_bits32(int32_t x)
{
    union {
        int32_t f;
        float t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double futrts_log64(double x)
{
    return log(x);
}
static inline double futrts_log2_64(double x)
{
    return log2(x);
}
static inline double futrts_log10_64(double x)
{
    return log10(x);
}
static inline double futrts_sqrt64(double x)
{
    return sqrt(x);
}
static inline double futrts_exp64(double x)
{
    return exp(x);
}
static inline double futrts_cos64(double x)
{
    return cos(x);
}
static inline double futrts_sin64(double x)
{
    return sin(x);
}
static inline double futrts_tan64(double x)
{
    return tan(x);
}
static inline double futrts_acos64(double x)
{
    return acos(x);
}
static inline double futrts_asin64(double x)
{
    return asin(x);
}
static inline double futrts_atan64(double x)
{
    return atan(x);
}
static inline double futrts_atan2_64(double x, double y)
{
    return atan2(x, y);
}
static inline double futrts_round64(double x)
{
    return rint(x);
}
static inline char futrts_isnan64(double x)
{
    return isnan(x);
}
static inline char futrts_isinf64(double x)
{
    return isinf(x);
}
static inline int64_t futrts_to_bits64(double x)
{
    union {
        double f;
        int64_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double futrts_from_bits64(int64_t x)
{
    union {
        int64_t f;
        double t;
    } p;
    
    p.f = x;
    return p.t;
}
static int futrts_main(struct futhark_context *ctx,
                       int64_t *out_out_memsizze_7923,
                       struct memblock_device *out_mem_p_7924,
                       int32_t *out_out_arrsizze_7925,
                       int64_t *out_out_memsizze_7926,
                       struct memblock_device *out_mem_p_7927,
                       int32_t *out_out_arrsizze_7928,
                       int64_t *out_out_memsizze_7929,
                       struct memblock_device *out_mem_p_7930,
                       int32_t *out_out_arrsizze_7931, int64_t a_mem_sizze_7843,
                       struct memblock_device a_mem_7844,
                       int64_t bs_mem_sizze_7845,
                       struct memblock_device bs_mem_7846, int32_t sizze_7709,
                       int32_t sizze_7710, int32_t o_7713)
{
    int64_t out_memsizze_7886;
    struct memblock_device out_mem_7885;
    
    out_mem_7885.references = NULL;
    
    int32_t out_arrsizze_7887;
    int64_t out_memsizze_7889;
    struct memblock_device out_mem_7888;
    
    out_mem_7888.references = NULL;
    
    int32_t out_arrsizze_7890;
    int64_t out_memsizze_7892;
    struct memblock_device out_mem_7891;
    
    out_mem_7891.references = NULL;
    
    int32_t out_arrsizze_7893;
    int32_t res_7714 = sdiv32(sizze_7710, o_7713);
    bool bounds_invalid_upwards_7715 = slt32(o_7713, 0);
    bool eq_x_zz_7716 = 0 == o_7713;
    bool not_p_7717 = !bounds_invalid_upwards_7715;
    bool p_and_eq_x_y_7718 = eq_x_zz_7716 && not_p_7717;
    bool dim_zzero_7719 = bounds_invalid_upwards_7715 || p_and_eq_x_y_7718;
    bool both_empty_7720 = eq_x_zz_7716 && dim_zzero_7719;
    bool empty_or_match_7724 = not_p_7717 || both_empty_7720;
    bool empty_or_match_cert_7725;
    
    if (!empty_or_match_7724) {
        ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%s\n",
                               "meyers.fut:12:1-37:22 -> meyers.fut:14:13-25 -> /futlib/array.fut:66:1-67:19",
                               "Function return value does not match shape of type ",
                               "*", "[", o_7713, "]", "t");
        if (memblock_unref_device(ctx, &out_mem_7891, "out_mem_7891") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_7888, "out_mem_7888") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_7885, "out_mem_7885") != 0)
            return 1;
        return 1;
    }
    
    int64_t binop_x_7848 = sext_i32_i64(o_7713);
    int64_t bytes_7847 = 4 * binop_x_7848;
    struct memblock_device mem_7849;
    
    mem_7849.references = NULL;
    if (memblock_alloc_device(ctx, &mem_7849, bytes_7847, "mem_7849"))
        return 1;
    
    int32_t group_sizze_7897;
    
    group_sizze_7897 = ctx->sizes.mainzigroup_sizze_7897;
    
    int32_t num_groups_7898;
    
    num_groups_7898 = squot32(o_7713 + sext_i32_i32(group_sizze_7897) - 1,
                              sext_i32_i32(group_sizze_7897));
    
    CUdeviceptr kernel_arg_7935 = mem_7849.mem;
    
    if ((((((1 && num_groups_7898 != 0) && 1 != 0) && 1 != 0) &&
          group_sizze_7897 != 0) && 1 != 0) && 1 != 0) {
        int perm[3] = {0, 1, 2};
        
        if (1 > 1 << 16) {
            perm[1] = perm[0];
            perm[0] = 1;
        }
        if (1 > 1 << 16) {
            perm[2] = perm[0];
            perm[0] = 2;
        }
        
        size_t grid[3];
        
        grid[perm[0]] = num_groups_7898;
        grid[perm[1]] = 1;
        grid[perm[2]] = 1;
        
        void *kernel_args_7932[] = {&sizze_7709, &o_7713, &kernel_arg_7935};
        int64_t time_start_7933 = 0, time_end_7934 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with grid size (", "replicate_7894");
            fprintf(stderr, "%d", num_groups_7898);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ") and block size (");
            fprintf(stderr, "%d", group_sizze_7897);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ").\n");
            time_start_7933 = get_wall_time();
        }
        CUDA_SUCCEED(cuLaunchKernel(ctx->replicate_7894, grid[0], grid[1],
                                    grid[2], group_sizze_7897, 1, 1, 0, NULL,
                                    kernel_args_7932, NULL));
        if (ctx->debugging) {
            CUDA_SUCCEED(cuCtxSynchronize());
            time_end_7934 = get_wall_time();
            fprintf(stderr, "Kernel %s runtime: %ldus\n", "replicate_7894",
                    time_end_7934 - time_start_7933);
        }
    }
    
    int64_t bytes_7850 = 8 * binop_x_7848;
    struct memblock_device mem_7852;
    
    mem_7852.references = NULL;
    if (memblock_alloc_device(ctx, &mem_7852, bytes_7850, "mem_7852"))
        return 1;
    
    int32_t group_sizze_7902;
    
    group_sizze_7902 = ctx->sizes.mainzigroup_sizze_7902;
    
    int32_t num_groups_7903;
    
    num_groups_7903 = squot32(o_7713 + sext_i32_i32(group_sizze_7902) - 1,
                              sext_i32_i32(group_sizze_7902));
    
    CUdeviceptr kernel_arg_7939 = mem_7852.mem;
    
    if ((((((1 && num_groups_7903 != 0) && 1 != 0) && 1 != 0) &&
          group_sizze_7902 != 0) && 1 != 0) && 1 != 0) {
        int perm[3] = {0, 1, 2};
        
        if (1 > 1 << 16) {
            perm[1] = perm[0];
            perm[0] = 1;
        }
        if (1 > 1 << 16) {
            perm[2] = perm[0];
            perm[0] = 2;
        }
        
        size_t grid[3];
        
        grid[perm[0]] = num_groups_7903;
        grid[perm[1]] = 1;
        grid[perm[2]] = 1;
        
        void *kernel_args_7936[] = {&o_7713, &kernel_arg_7939};
        int64_t time_start_7937 = 0, time_end_7938 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with grid size (", "replicate_7899");
            fprintf(stderr, "%d", num_groups_7903);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ") and block size (");
            fprintf(stderr, "%d", group_sizze_7902);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ").\n");
            time_start_7937 = get_wall_time();
        }
        CUDA_SUCCEED(cuLaunchKernel(ctx->replicate_7899, grid[0], grid[1],
                                    grid[2], group_sizze_7902, 1, 1, 0, NULL,
                                    kernel_args_7936, NULL));
        if (ctx->debugging) {
            CUDA_SUCCEED(cuCtxSynchronize());
            time_end_7938 = get_wall_time();
            fprintf(stderr, "Kernel %s runtime: %ldus\n", "replicate_7899",
                    time_end_7938 - time_start_7937);
        }
    }
    
    struct memblock_device mem_7855;
    
    mem_7855.references = NULL;
    if (memblock_alloc_device(ctx, &mem_7855, bytes_7850, "mem_7855"))
        return 1;
    
    int32_t group_sizze_7907;
    
    group_sizze_7907 = ctx->sizes.mainzigroup_sizze_7907;
    
    int32_t num_groups_7908;
    
    num_groups_7908 = squot32(o_7713 + sext_i32_i32(group_sizze_7907) - 1,
                              sext_i32_i32(group_sizze_7907));
    
    CUdeviceptr kernel_arg_7943 = mem_7855.mem;
    
    if ((((((1 && num_groups_7908 != 0) && 1 != 0) && 1 != 0) &&
          group_sizze_7907 != 0) && 1 != 0) && 1 != 0) {
        int perm[3] = {0, 1, 2};
        
        if (1 > 1 << 16) {
            perm[1] = perm[0];
            perm[0] = 1;
        }
        if (1 > 1 << 16) {
            perm[2] = perm[0];
            perm[0] = 2;
        }
        
        size_t grid[3];
        
        grid[perm[0]] = num_groups_7908;
        grid[perm[1]] = 1;
        grid[perm[2]] = 1;
        
        void *kernel_args_7940[] = {&o_7713, &kernel_arg_7943};
        int64_t time_start_7941 = 0, time_end_7942 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with grid size (", "replicate_7904");
            fprintf(stderr, "%d", num_groups_7908);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ") and block size (");
            fprintf(stderr, "%d", group_sizze_7907);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ").\n");
            time_start_7941 = get_wall_time();
        }
        CUDA_SUCCEED(cuLaunchKernel(ctx->replicate_7904, grid[0], grid[1],
                                    grid[2], group_sizze_7907, 1, 1, 0, NULL,
                                    kernel_args_7940, NULL));
        if (ctx->debugging) {
            CUDA_SUCCEED(cuCtxSynchronize());
            time_end_7942 = get_wall_time();
            fprintf(stderr, "Kernel %s runtime: %ldus\n", "replicate_7904",
                    time_end_7942 - time_start_7941);
        }
    }
    
    int64_t arg_7729 = zext_i32_i64(sizze_7709);
    int64_t y_7730 = arg_7729 - 1;
    int64_t res_7731 = 1 << y_7730;
    bool empty_or_match_cert_7732;
    
    if (!empty_or_match_7724) {
        ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%s\n",
                               "meyers.fut:12:1-37:22 -> meyers.fut:18:15-20 -> /futlib/array.fut:61:1-62:12",
                               "Function return value does not match shape of type ",
                               "*", "[", o_7713, "]", "intrinsics.i32");
        if (memblock_unref_device(ctx, &mem_7855, "mem_7855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_7852, "mem_7852") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_7849, "mem_7849") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_7891, "out_mem_7891") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_7888, "out_mem_7888") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_7885, "out_mem_7885") != 0)
            return 1;
        return 1;
    }
    
    struct memblock_device mem_7858;
    
    mem_7858.references = NULL;
    if (memblock_alloc_device(ctx, &mem_7858, 2048, "mem_7858"))
        return 1;
    
    int32_t group_sizze_7912;
    
    group_sizze_7912 = ctx->sizes.mainzigroup_sizze_7912;
    
    int32_t num_groups_7913;
    
    num_groups_7913 = squot32(256 + sext_i32_i32(group_sizze_7912) - 1,
                              sext_i32_i32(group_sizze_7912));
    
    CUdeviceptr kernel_arg_7947 = mem_7858.mem;
    
    if ((((((1 && num_groups_7913 != 0) && 1 != 0) && 1 != 0) &&
          group_sizze_7912 != 0) && 1 != 0) && 1 != 0) {
        int perm[3] = {0, 1, 2};
        
        if (1 > 1 << 16) {
            perm[1] = perm[0];
            perm[0] = 1;
        }
        if (1 > 1 << 16) {
            perm[2] = perm[0];
            perm[0] = 2;
        }
        
        size_t grid[3];
        
        grid[perm[0]] = num_groups_7913;
        grid[perm[1]] = 1;
        grid[perm[2]] = 1;
        
        void *kernel_args_7944[] = {&kernel_arg_7947};
        int64_t time_start_7945 = 0, time_end_7946 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with grid size (", "replicate_7909");
            fprintf(stderr, "%d", num_groups_7913);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ") and block size (");
            fprintf(stderr, "%d", group_sizze_7912);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ").\n");
            time_start_7945 = get_wall_time();
        }
        CUDA_SUCCEED(cuLaunchKernel(ctx->replicate_7909, grid[0], grid[1],
                                    grid[2], group_sizze_7912, 1, 1, 0, NULL,
                                    kernel_args_7944, NULL));
        if (ctx->debugging) {
            CUDA_SUCCEED(cuCtxSynchronize());
            time_end_7946 = get_wall_time();
            fprintf(stderr, "Kernel %s runtime: %ldus\n", "replicate_7909",
                    time_end_7946 - time_start_7945);
        }
    }
    for (int32_t i_7737 = 0; i_7737 < sizze_7709; i_7737++) {
        int16_t read_res_7948;
        
        CUDA_SUCCEED(cuMemcpyDtoH(&read_res_7948, a_mem_7844.mem + i_7737 * 2,
                                  sizeof(int16_t)));
        
        int16_t arg_7742 = read_res_7948;
        int32_t res_7743 = zext_i16_i32(arg_7742);
        bool x_7744 = sle32(0, res_7743);
        bool y_7745 = slt32(res_7743, 256);
        bool bounds_check_7746 = x_7744 && y_7745;
        bool index_certs_7747;
        
        if (!bounds_check_7746) {
            ctx->error = msgprintf("Error at %s:\n%s%d%s%d%s\n",
                                   "meyers.fut:12:1-37:22 -> meyers.fut:23:27-36",
                                   "Index [", res_7743,
                                   "] out of bounds for array of shape [", 256,
                                   "].");
            if (memblock_unref_device(ctx, &mem_7858, "mem_7858") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_7855, "mem_7855") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_7852, "mem_7852") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_7849, "mem_7849") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_7891, "out_mem_7891") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_7888, "out_mem_7888") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_7885, "out_mem_7885") != 0)
                return 1;
            return 1;
        }
        
        int64_t read_res_7949;
        
        CUDA_SUCCEED(cuMemcpyDtoH(&read_res_7949, mem_7858.mem + res_7743 * 8,
                                  sizeof(int64_t)));
        
        int64_t x_7748 = read_res_7949;
        int64_t arg_7749 = zext_i32_i64(i_7737);
        int64_t y_7750 = 1 << arg_7749;
        int64_t lw_val_7751 = x_7748 | y_7750;
        int64_t write_tmp_7950 = lw_val_7751;
        
        CUDA_SUCCEED(cuMemcpyHtoD(mem_7858.mem + res_7743 * 8, &write_tmp_7950,
                                  sizeof(int64_t)));
    }
    
    bool loop_nonempty_7840 = slt32(0, res_7714);
    int32_t group_sizze_7798;
    
    group_sizze_7798 = ctx->sizes.mainzigroup_sizze_7797;
    
    int32_t y_7799 = group_sizze_7798 - 1;
    int32_t x_7800 = o_7713 + y_7799;
    int32_t num_groups_7801;
    
    if (loop_nonempty_7840) {
        int32_t x_7841 = squot32(x_7800, group_sizze_7798);
        
        num_groups_7801 = x_7841;
    } else {
        num_groups_7801 = 0;
    }
    
    int32_t num_threads_7802 = group_sizze_7798 * num_groups_7801;
    struct memblock_device res_mem_7879;
    
    res_mem_7879.references = NULL;
    
    struct memblock_device res_mem_7881;
    
    res_mem_7881.references = NULL;
    
    struct memblock_device res_mem_7883;
    
    res_mem_7883.references = NULL;
    
    struct memblock_device scs_mem_7864;
    
    scs_mem_7864.references = NULL;
    
    struct memblock_device mvs_mem_7866;
    
    mvs_mem_7866.references = NULL;
    
    struct memblock_device pvs_mem_7868;
    
    pvs_mem_7868.references = NULL;
    if (memblock_set_device(ctx, &scs_mem_7864, &mem_7849, "mem_7849") != 0)
        return 1;
    if (memblock_set_device(ctx, &mvs_mem_7866, &mem_7852, "mem_7852") != 0)
        return 1;
    if (memblock_set_device(ctx, &pvs_mem_7868, &mem_7855, "mem_7855") != 0)
        return 1;
    for (int32_t i_7759 = 0; i_7759 < res_7714; i_7759++) {
        struct memblock_device mem_7871;
        
        mem_7871.references = NULL;
        if (memblock_alloc_device(ctx, &mem_7871, bytes_7850, "mem_7871"))
            return 1;
        
        struct memblock_device mem_7874;
        
        mem_7874.references = NULL;
        if (memblock_alloc_device(ctx, &mem_7874, bytes_7850, "mem_7874"))
            return 1;
        
        struct memblock_device mem_7877;
        
        mem_7877.references = NULL;
        if (memblock_alloc_device(ctx, &mem_7877, bytes_7847, "mem_7877"))
            return 1;
        
        CUdeviceptr kernel_arg_7954 = bs_mem_7846.mem;
        CUdeviceptr kernel_arg_7955 = mem_7858.mem;
        CUdeviceptr kernel_arg_7956 = scs_mem_7864.mem;
        CUdeviceptr kernel_arg_7957 = mvs_mem_7866.mem;
        CUdeviceptr kernel_arg_7958 = pvs_mem_7868.mem;
        CUdeviceptr kernel_arg_7959 = mem_7871.mem;
        CUdeviceptr kernel_arg_7960 = mem_7874.mem;
        CUdeviceptr kernel_arg_7961 = mem_7877.mem;
        
        if ((((((1 && num_groups_7801 != 0) && 1 != 0) && 1 != 0) &&
              group_sizze_7798 != 0) && 1 != 0) && 1 != 0) {
            int perm[3] = {0, 1, 2};
            
            if (1 > 1 << 16) {
                perm[1] = perm[0];
                perm[0] = 1;
            }
            if (1 > 1 << 16) {
                perm[2] = perm[0];
                perm[0] = 2;
            }
            
            size_t grid[3];
            
            grid[perm[0]] = num_groups_7801;
            grid[perm[1]] = 1;
            grid[perm[2]] = 1;
            
            void *kernel_args_7951[] = {&o_7713, &res_7714, &res_7731, &i_7759,
                                        &kernel_arg_7954, &kernel_arg_7955,
                                        &kernel_arg_7956, &kernel_arg_7957,
                                        &kernel_arg_7958, &kernel_arg_7959,
                                        &kernel_arg_7960, &kernel_arg_7961};
            int64_t time_start_7952 = 0, time_end_7953 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with grid size (", "map_7803");
                fprintf(stderr, "%d", num_groups_7801);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ") and block size (");
                fprintf(stderr, "%d", group_sizze_7798);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ").\n");
                time_start_7952 = get_wall_time();
            }
            CUDA_SUCCEED(cuLaunchKernel(ctx->map_7803, grid[0], grid[1],
                                        grid[2], group_sizze_7798, 1, 1, 0,
                                        NULL, kernel_args_7951, NULL));
            if (ctx->debugging) {
                CUDA_SUCCEED(cuCtxSynchronize());
                time_end_7953 = get_wall_time();
                fprintf(stderr, "Kernel %s runtime: %ldus\n", "map_7803",
                        time_end_7953 - time_start_7952);
            }
        }
        
        struct memblock_device scs_mem_tmp_7915;
        
        scs_mem_tmp_7915.references = NULL;
        if (memblock_set_device(ctx, &scs_mem_tmp_7915, &mem_7877,
                                "mem_7877") != 0)
            return 1;
        
        struct memblock_device mvs_mem_tmp_7916;
        
        mvs_mem_tmp_7916.references = NULL;
        if (memblock_set_device(ctx, &mvs_mem_tmp_7916, &mem_7871,
                                "mem_7871") != 0)
            return 1;
        
        struct memblock_device pvs_mem_tmp_7917;
        
        pvs_mem_tmp_7917.references = NULL;
        if (memblock_set_device(ctx, &pvs_mem_tmp_7917, &mem_7874,
                                "mem_7874") != 0)
            return 1;
        if (memblock_set_device(ctx, &scs_mem_7864, &scs_mem_tmp_7915,
                                "scs_mem_tmp_7915") != 0)
            return 1;
        if (memblock_set_device(ctx, &mvs_mem_7866, &mvs_mem_tmp_7916,
                                "mvs_mem_tmp_7916") != 0)
            return 1;
        if (memblock_set_device(ctx, &pvs_mem_7868, &pvs_mem_tmp_7917,
                                "pvs_mem_tmp_7917") != 0)
            return 1;
        if (memblock_unref_device(ctx, &pvs_mem_tmp_7917, "pvs_mem_tmp_7917") !=
            0)
            return 1;
        if (memblock_unref_device(ctx, &mvs_mem_tmp_7916, "mvs_mem_tmp_7916") !=
            0)
            return 1;
        if (memblock_unref_device(ctx, &scs_mem_tmp_7915, "scs_mem_tmp_7915") !=
            0)
            return 1;
        if (memblock_unref_device(ctx, &mem_7877, "mem_7877") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_7874, "mem_7874") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_7871, "mem_7871") != 0)
            return 1;
    }
    if (memblock_set_device(ctx, &res_mem_7879, &scs_mem_7864,
                            "scs_mem_7864") != 0)
        return 1;
    if (memblock_set_device(ctx, &res_mem_7881, &mvs_mem_7866,
                            "mvs_mem_7866") != 0)
        return 1;
    if (memblock_set_device(ctx, &res_mem_7883, &pvs_mem_7868,
                            "pvs_mem_7868") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_7849, "mem_7849") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_7852, "mem_7852") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_7855, "mem_7855") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_7858, "mem_7858") != 0)
        return 1;
    out_arrsizze_7887 = o_7713;
    out_arrsizze_7890 = o_7713;
    out_arrsizze_7893 = o_7713;
    out_memsizze_7886 = bytes_7847;
    if (memblock_set_device(ctx, &out_mem_7885, &res_mem_7879,
                            "res_mem_7879") != 0)
        return 1;
    out_memsizze_7889 = bytes_7850;
    if (memblock_set_device(ctx, &out_mem_7888, &res_mem_7881,
                            "res_mem_7881") != 0)
        return 1;
    out_memsizze_7892 = bytes_7850;
    if (memblock_set_device(ctx, &out_mem_7891, &res_mem_7883,
                            "res_mem_7883") != 0)
        return 1;
    *out_out_memsizze_7923 = out_memsizze_7886;
    (*out_mem_p_7924).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_7924, &out_mem_7885,
                            "out_mem_7885") != 0)
        return 1;
    *out_out_arrsizze_7925 = out_arrsizze_7887;
    *out_out_memsizze_7926 = out_memsizze_7889;
    (*out_mem_p_7927).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_7927, &out_mem_7888,
                            "out_mem_7888") != 0)
        return 1;
    *out_out_arrsizze_7928 = out_arrsizze_7890;
    *out_out_memsizze_7929 = out_memsizze_7892;
    (*out_mem_p_7930).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_7930, &out_mem_7891,
                            "out_mem_7891") != 0)
        return 1;
    *out_out_arrsizze_7931 = out_arrsizze_7893;
    if (memblock_unref_device(ctx, &pvs_mem_7868, "pvs_mem_7868") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mvs_mem_7866, "mvs_mem_7866") != 0)
        return 1;
    if (memblock_unref_device(ctx, &scs_mem_7864, "scs_mem_7864") != 0)
        return 1;
    if (memblock_unref_device(ctx, &res_mem_7883, "res_mem_7883") != 0)
        return 1;
    if (memblock_unref_device(ctx, &res_mem_7881, "res_mem_7881") != 0)
        return 1;
    if (memblock_unref_device(ctx, &res_mem_7879, "res_mem_7879") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_7858, "mem_7858") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_7855, "mem_7855") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_7852, "mem_7852") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_7849, "mem_7849") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_7891, "out_mem_7891") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_7888, "out_mem_7888") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_7885, "out_mem_7885") != 0)
        return 1;
    return 0;
}
struct futhark_u64_1d {
    struct memblock_device mem;
    int64_t shape[1];
} ;
struct futhark_u64_1d *futhark_new_u64_1d(struct futhark_context *ctx,
                                          uint64_t *data, int dim0)
{
    struct futhark_u64_1d *bad = NULL;
    struct futhark_u64_1d *arr = malloc(sizeof(struct futhark_u64_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, dim0 * sizeof(uint64_t),
                              "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    CUDA_SUCCEED(cuMemcpyHtoD(arr->mem.mem + 0, data + 0, dim0 *
                              sizeof(uint64_t)));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_u64_1d *futhark_new_raw_u64_1d(struct futhark_context *ctx,
                                              CUdeviceptr data, int offset,
                                              int dim0)
{
    struct futhark_u64_1d *bad = NULL;
    struct futhark_u64_1d *arr = malloc(sizeof(struct futhark_u64_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, dim0 * sizeof(uint64_t),
                              "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    CUDA_SUCCEED(cuMemcpy(arr->mem.mem + 0, data + offset, dim0 *
                          sizeof(uint64_t)));
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_u64_1d(struct futhark_context *ctx, struct futhark_u64_1d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref_device(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_u64_1d(struct futhark_context *ctx,
                          struct futhark_u64_1d *arr, uint64_t *data)
{
    lock_lock(&ctx->lock);
    CUDA_SUCCEED(cuMemcpyDtoH(data + 0, arr->mem.mem + 0, arr->shape[0] *
                              sizeof(uint64_t)));
    lock_unlock(&ctx->lock);
    return 0;
}
CUdeviceptr futhark_values_raw_u64_1d(struct futhark_context *ctx,
                                      struct futhark_u64_1d *arr)
{
    return arr->mem.mem;
}
int64_t *futhark_shape_u64_1d(struct futhark_context *ctx,
                              struct futhark_u64_1d *arr)
{
    return arr->shape;
}
struct futhark_i32_1d {
    struct memblock_device mem;
    int64_t shape[1];
} ;
struct futhark_i32_1d *futhark_new_i32_1d(struct futhark_context *ctx,
                                          int32_t *data, int dim0)
{
    struct futhark_i32_1d *bad = NULL;
    struct futhark_i32_1d *arr = malloc(sizeof(struct futhark_i32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, dim0 * sizeof(int32_t),
                              "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    CUDA_SUCCEED(cuMemcpyHtoD(arr->mem.mem + 0, data + 0, dim0 *
                              sizeof(int32_t)));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_i32_1d *futhark_new_raw_i32_1d(struct futhark_context *ctx,
                                              CUdeviceptr data, int offset,
                                              int dim0)
{
    struct futhark_i32_1d *bad = NULL;
    struct futhark_i32_1d *arr = malloc(sizeof(struct futhark_i32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, dim0 * sizeof(int32_t),
                              "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    CUDA_SUCCEED(cuMemcpy(arr->mem.mem + 0, data + offset, dim0 *
                          sizeof(int32_t)));
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_i32_1d(struct futhark_context *ctx, struct futhark_i32_1d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref_device(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_i32_1d(struct futhark_context *ctx,
                          struct futhark_i32_1d *arr, int32_t *data)
{
    lock_lock(&ctx->lock);
    CUDA_SUCCEED(cuMemcpyDtoH(data + 0, arr->mem.mem + 0, arr->shape[0] *
                              sizeof(int32_t)));
    lock_unlock(&ctx->lock);
    return 0;
}
CUdeviceptr futhark_values_raw_i32_1d(struct futhark_context *ctx,
                                      struct futhark_i32_1d *arr)
{
    return arr->mem.mem;
}
int64_t *futhark_shape_i32_1d(struct futhark_context *ctx,
                              struct futhark_i32_1d *arr)
{
    return arr->shape;
}
struct futhark_u16_1d {
    struct memblock_device mem;
    int64_t shape[1];
} ;
struct futhark_u16_1d *futhark_new_u16_1d(struct futhark_context *ctx,
                                          uint16_t *data, int dim0)
{
    struct futhark_u16_1d *bad = NULL;
    struct futhark_u16_1d *arr = malloc(sizeof(struct futhark_u16_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, dim0 * sizeof(uint16_t),
                              "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    CUDA_SUCCEED(cuMemcpyHtoD(arr->mem.mem + 0, data + 0, dim0 *
                              sizeof(uint16_t)));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_u16_1d *futhark_new_raw_u16_1d(struct futhark_context *ctx,
                                              CUdeviceptr data, int offset,
                                              int dim0)
{
    struct futhark_u16_1d *bad = NULL;
    struct futhark_u16_1d *arr = malloc(sizeof(struct futhark_u16_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, dim0 * sizeof(uint16_t),
                              "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    CUDA_SUCCEED(cuMemcpy(arr->mem.mem + 0, data + offset, dim0 *
                          sizeof(uint16_t)));
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_u16_1d(struct futhark_context *ctx, struct futhark_u16_1d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref_device(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_u16_1d(struct futhark_context *ctx,
                          struct futhark_u16_1d *arr, uint16_t *data)
{
    lock_lock(&ctx->lock);
    CUDA_SUCCEED(cuMemcpyDtoH(data + 0, arr->mem.mem + 0, arr->shape[0] *
                              sizeof(uint16_t)));
    lock_unlock(&ctx->lock);
    return 0;
}
CUdeviceptr futhark_values_raw_u16_1d(struct futhark_context *ctx,
                                      struct futhark_u16_1d *arr)
{
    return arr->mem.mem;
}
int64_t *futhark_shape_u16_1d(struct futhark_context *ctx,
                              struct futhark_u16_1d *arr)
{
    return arr->shape;
}
int futhark_entry_main(struct futhark_context *ctx,
                       struct futhark_i32_1d **out0,
                       struct futhark_u64_1d **out1,
                       struct futhark_u64_1d **out2, const
                       struct futhark_u16_1d *in0, const
                       struct futhark_u16_1d *in1, const int32_t in2)
{
    int64_t a_mem_sizze_7843;
    struct memblock_device a_mem_7844;
    
    a_mem_7844.references = NULL;
    
    int64_t bs_mem_sizze_7845;
    struct memblock_device bs_mem_7846;
    
    bs_mem_7846.references = NULL;
    
    int32_t sizze_7709;
    int32_t sizze_7710;
    int32_t o_7713;
    int64_t out_memsizze_7886;
    struct memblock_device out_mem_7885;
    
    out_mem_7885.references = NULL;
    
    int32_t out_arrsizze_7887;
    int64_t out_memsizze_7889;
    struct memblock_device out_mem_7888;
    
    out_mem_7888.references = NULL;
    
    int32_t out_arrsizze_7890;
    int64_t out_memsizze_7892;
    struct memblock_device out_mem_7891;
    
    out_mem_7891.references = NULL;
    
    int32_t out_arrsizze_7893;
    
    lock_lock(&ctx->lock);
    a_mem_7844 = in0->mem;
    a_mem_sizze_7843 = in0->mem.size;
    sizze_7709 = in0->shape[0];
    bs_mem_7846 = in1->mem;
    bs_mem_sizze_7845 = in1->mem.size;
    sizze_7710 = in1->shape[0];
    o_7713 = in2;
    
    int ret = futrts_main(ctx, &out_memsizze_7886, &out_mem_7885,
                          &out_arrsizze_7887, &out_memsizze_7889, &out_mem_7888,
                          &out_arrsizze_7890, &out_memsizze_7892, &out_mem_7891,
                          &out_arrsizze_7893, a_mem_sizze_7843, a_mem_7844,
                          bs_mem_sizze_7845, bs_mem_7846, sizze_7709,
                          sizze_7710, o_7713);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_i32_1d))) != NULL);
        (*out0)->mem = out_mem_7885;
        (*out0)->shape[0] = out_arrsizze_7887;
        assert((*out1 = malloc(sizeof(struct futhark_u64_1d))) != NULL);
        (*out1)->mem = out_mem_7888;
        (*out1)->shape[0] = out_arrsizze_7890;
        assert((*out2 = malloc(sizeof(struct futhark_u64_1d))) != NULL);
        (*out2)->mem = out_mem_7891;
        (*out2)->shape[0] = out_arrsizze_7893;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
