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

struct futhark_i32_2d ;
struct futhark_i32_2d *futhark_new_i32_2d(struct futhark_context *ctx,
                                          int32_t *data, int dim0, int dim1);
struct futhark_i32_2d *futhark_new_raw_i32_2d(struct futhark_context *ctx,
                                              CUdeviceptr data, int offset,
                                              int dim0, int dim1);
int futhark_free_i32_2d(struct futhark_context *ctx,
                        struct futhark_i32_2d *arr);
int futhark_values_i32_2d(struct futhark_context *ctx,
                          struct futhark_i32_2d *arr, int32_t *data);
CUdeviceptr futhark_values_raw_i32_2d(struct futhark_context *ctx,
                                      struct futhark_i32_2d *arr);
int64_t *futhark_shape_i32_2d(struct futhark_context *ctx,
                              struct futhark_i32_2d *arr);
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

/*
 * Opaque values
*/


/*
 * Entry points
*/

int futhark_entry__myers(struct futhark_context *ctx,
                         struct futhark_u16_1d **out0, const
                         struct futhark_u16_1d *in0, const
                         struct futhark_u16_1d *in1, const int32_t in2);
int futhark_entry__format(struct futhark_context *ctx,
                          struct futhark_u16_1d **out0, const
                          struct futhark_i32_2d *in0);

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
static void futrts_cli_entry__myers(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    struct futhark_u16_1d *read_value_8469;
    int64_t read_shape_8470[1];
    int16_t *read_arr_8471 = NULL;
    
    errno = 0;
    if (read_array(&u16_info, (void **) &read_arr_8471, read_shape_8470, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[]",
              u16_info.type_name, strerror(errno));
    
    struct futhark_u16_1d *read_value_8472;
    int64_t read_shape_8473[1];
    int16_t *read_arr_8474 = NULL;
    
    errno = 0;
    if (read_array(&u16_info, (void **) &read_arr_8474, read_shape_8473, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[]",
              u16_info.type_name, strerror(errno));
    
    int32_t read_value_8475;
    
    if (read_scalar(&i32_info, &read_value_8475) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 2,
              i32_info.type_name, strerror(errno));
    
    struct futhark_u16_1d *result_8476;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_8469 = futhark_new_u16_1d(ctx, read_arr_8471,
                                                     read_shape_8470[0])) != 0);
        assert((read_value_8472 = futhark_new_u16_1d(ctx, read_arr_8474,
                                                     read_shape_8473[0])) != 0);
        ;
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry__myers(ctx, &result_8476, read_value_8469,
                                 read_value_8472, read_value_8475);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_u16_1d(ctx, read_value_8469) == 0);
        assert(futhark_free_u16_1d(ctx, read_value_8472) == 0);
        ;
        assert(futhark_free_u16_1d(ctx, result_8476) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_8469 = futhark_new_u16_1d(ctx, read_arr_8471,
                                                     read_shape_8470[0])) != 0);
        assert((read_value_8472 = futhark_new_u16_1d(ctx, read_arr_8474,
                                                     read_shape_8473[0])) != 0);
        ;
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry__myers(ctx, &result_8476, read_value_8469,
                                 read_value_8472, read_value_8475);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_u16_1d(ctx, read_value_8469) == 0);
        assert(futhark_free_u16_1d(ctx, read_value_8472) == 0);
        ;
        if (run < num_runs - 1) {
            assert(futhark_free_u16_1d(ctx, result_8476) == 0);
        }
    }
    free(read_arr_8471);
    free(read_arr_8474);
    ;
    if (binary_output)
        set_binary_mode(stdout);
    {
        int16_t *arr = calloc(sizeof(int16_t), futhark_shape_u16_1d(ctx,
                                                                    result_8476)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_u16_1d(ctx, result_8476, arr) == 0);
        write_array(stdout, binary_output, &u16_info, arr,
                    futhark_shape_u16_1d(ctx, result_8476), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_u16_1d(ctx, result_8476) == 0);
}
static void futrts_cli_entry__format(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    struct futhark_i32_2d *read_value_8477;
    int64_t read_shape_8478[2];
    int32_t *read_arr_8479 = NULL;
    
    errno = 0;
    if (read_array(&i32_info, (void **) &read_arr_8479, read_shape_8478, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[][]",
              i32_info.type_name, strerror(errno));
    
    struct futhark_u16_1d *result_8480;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_8477 = futhark_new_i32_2d(ctx, read_arr_8479,
                                                     read_shape_8478[0],
                                                     read_shape_8478[1])) != 0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry__format(ctx, &result_8480, read_value_8477);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_i32_2d(ctx, read_value_8477) == 0);
        assert(futhark_free_u16_1d(ctx, result_8480) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_8477 = futhark_new_i32_2d(ctx, read_arr_8479,
                                                     read_shape_8478[0],
                                                     read_shape_8478[1])) != 0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry__format(ctx, &result_8480, read_value_8477);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_i32_2d(ctx, read_value_8477) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_u16_1d(ctx, result_8480) == 0);
        }
    }
    free(read_arr_8479);
    if (binary_output)
        set_binary_mode(stdout);
    {
        int16_t *arr = calloc(sizeof(int16_t), futhark_shape_u16_1d(ctx,
                                                                    result_8480)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_u16_1d(ctx, result_8480, arr) == 0);
        write_array(stdout, binary_output, &u16_info, arr,
                    futhark_shape_u16_1d(ctx, result_8480), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_u16_1d(ctx, result_8480) == 0);
}
typedef void entry_point_fun(struct futhark_context *);
struct entry_point_entry {
    const char *name;
    entry_point_fun *fun;
} ;
int main(int argc, char **argv)
{
    fut_progname = argv[0];
    
    struct entry_point_entry entry_points[] = {{.name ="_myers", .fun =
                                                futrts_cli_entry__myers},
                                               {.name ="_format", .fun =
                                                futrts_cli_entry__format}};
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
            "_f64(double x)\n{\n    return x;\n}\n__kernel void map_8297(int32_t sizze_8184, int32_t sizze_8185,\n                       int32_t flat_dim_8188, __global\n                       unsigned char *x_mem_8349, __global\n                       unsigned char *mem_8352)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t wave_sizze_8394;\n    int32_t group_sizze_8395;\n    int32_t gtid_8290;\n    int32_t global_tid_8297;\n    int32_t local_tid_8298;\n    int32_t group_id_8299;\n    \n    global_tid_8297 = get_global_id(0);\n    local_tid_8298 = get_local_id(0);\n    group_sizze_8395 = get_local_size(0);\n    wave_sizze_8394 = LOCKSTEP_WIDTH;\n    group_id_8299 = get_group_id(0);\n    gtid_8290 = global_tid_8297;\n    \n    int32_t new_index_8344;\n    int32_t binop_y_8346;\n    int32_t new_index_8347;\n    int32_t x_8300;\n    int16_t arg_8301;\n    \n    if (slt32(gtid_8290, flat_dim_8188)) {\n        new_index_8344 = squot32(gtid_8290, sizze_8184);\n        binop_y_8346 = sizze_8184 * new_index_8344;\n        new_index_8347 = gtid_8290 - binop_y_8346;\n        x_8300 = *(__global int32_t *) &x_mem_8349[(new_index_8347 *\n                                                    sizze_8185 +\n                                                    new_index_8344) * 4];\n        arg_8301 = zext_i32_i16(x_8300);\n    }\n    if (slt32(gtid_8290, flat_dim_8188)) {\n        *(__global int16_t *) &mem_8352[gtid_8290 * 2] = arg_8301;\n    }\n}\n__kernel void map_8309(int32_t x_8197, int64_t res_8200,\n                       int32_t index_primexp_8254, __global\n                       unsigned char *x_mem_8351, __global\n                       unsigned char *mem_8363, __global\n                       unsigned char *scs_mem_8369, __global\n                       unsigned char *mvs_mem_8371, __global\n                       unsigned char *pvs_mem_8373, __global\n                       unsigned char *mem_8376, __global\n                       unsigned char *mem_8379, __global\n           ",
            "            unsigned char *mem_8382)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t wave_sizze_8426;\n    int32_t group_sizze_8427;\n    int32_t gtid_8302;\n    int32_t global_tid_8309;\n    int32_t local_tid_8310;\n    int32_t group_id_8311;\n    \n    global_tid_8309 = get_global_id(0);\n    local_tid_8310 = get_local_id(0);\n    group_sizze_8427 = get_local_size(0);\n    wave_sizze_8426 = LOCKSTEP_WIDTH;\n    group_id_8311 = get_group_id(0);\n    gtid_8302 = global_tid_8309;\n    \n    int64_t x_8313;\n    int64_t x_8314;\n    int16_t x_8315;\n    int32_t i_8316;\n    int16_t arg_8317;\n    int32_t res_8318;\n    int64_t res_8319;\n    int64_t res_8320;\n    int64_t x_8321;\n    int64_t x_8322;\n    int64_t x_8323;\n    int64_t res_8324;\n    int64_t complement_arg_8325;\n    int64_t y_8326;\n    int64_t res_8327;\n    int64_t res_8328;\n    int64_t arg_8329;\n    bool res_8330;\n    int16_t res_8331;\n    int64_t x_8337;\n    int64_t res_8338;\n    int64_t x_8339;\n    int64_t complement_arg_8340;\n    int64_t y_8341;\n    int64_t res_8342;\n    int64_t res_8343;\n    \n    if (slt32(gtid_8302, x_8197)) {\n        x_8313 = *(__global int64_t *) &mvs_mem_8371[gtid_8302 * 8];\n        x_8314 = *(__global int64_t *) &pvs_mem_8373[gtid_8302 * 8];\n        x_8315 = *(__global int16_t *) &scs_mem_8369[gtid_8302 * 2];\n        i_8316 = index_primexp_8254 + gtid_8302;\n        arg_8317 = *(__global int16_t *) &x_mem_8351[i_8316 * 2];\n        res_8318 = zext_i16_i32(arg_8317);\n        res_8319 = *(__global int64_t *) &mem_8363[res_8318 * 8];\n        res_8320 = x_8313 | res_8319;\n        x_8321 = x_8314 & res_8319;\n        x_8322 = x_8314 + x_8321;\n        x_8323 = x_8314 ^ x_8322;\n        res_8324 = res_8319 | x_8323;\n        complement_arg_8325 = x_8313 | x_8314;\n        y_8326 = ~complement_arg_8325;\n        res_8327 = res_8324 | y_8326;\n        res_8328 = x_8314 & res_8324;\n        arg_8329 = res_8200 & res_8327;\n        res_8330 = itob_i64_bool(arg_8329);\n",
            "        if (res_8330) {\n            int16_t res_8332 = 1 + x_8315;\n            \n            res_8331 = res_8332;\n        } else {\n            int64_t arg_8333;\n            bool res_8334;\n            int16_t res_8335;\n            \n            arg_8333 = res_8200 & res_8328;\n            res_8334 = itob_i64_bool(arg_8333);\n            if (res_8334) {\n                int16_t res_8336 = x_8315 - 1;\n                \n                res_8335 = res_8336;\n            } else {\n                res_8335 = x_8315;\n            }\n            res_8331 = res_8335;\n        }\n        x_8337 = res_8327 << 1;\n        res_8338 = 1 | x_8337;\n        x_8339 = res_8328 << 1;\n        complement_arg_8340 = res_8320 | res_8338;\n        y_8341 = ~complement_arg_8340;\n        res_8342 = x_8339 | y_8341;\n        res_8343 = res_8320 & res_8338;\n    }\n    if (slt32(gtid_8302, x_8197)) {\n        *(__global int64_t *) &mem_8376[gtid_8302 * 8] = res_8343;\n    }\n    if (slt32(gtid_8302, x_8197)) {\n        *(__global int64_t *) &mem_8379[gtid_8302 * 8] = res_8342;\n    }\n    if (slt32(gtid_8302, x_8197)) {\n        *(__global int16_t *) &mem_8382[gtid_8302 * 2] = res_8331;\n    }\n}\n__kernel void replicate_8399(int32_t x_8197, int16_t arg_8201, __global\n                             unsigned char *mem_8354)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t replicate_gtid_8399;\n    int32_t replicate_ltid_8400;\n    int32_t replicate_gid_8401;\n    \n    replicate_gtid_8399 = get_global_id(0);\n    replicate_ltid_8400 = get_local_id(0);\n    replicate_gid_8401 = get_group_id(0);\n    if (slt32(replicate_gtid_8399, x_8197)) {\n        *(__global int16_t *) &mem_8354[replicate_gtid_8399 * 2] = arg_8201;\n    }\n}\n__kernel void replicate_8404(int32_t x_8197, __global unsigned char *mem_8357)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t replicate_gtid_8404;\n    int32_t replicate_ltid_8405;\n    int32_t replicate_",
            "gid_8406;\n    \n    replicate_gtid_8404 = get_global_id(0);\n    replicate_ltid_8405 = get_local_id(0);\n    replicate_gid_8406 = get_group_id(0);\n    if (slt32(replicate_gtid_8404, x_8197)) {\n        *(__global int64_t *) &mem_8357[replicate_gtid_8404 * 8] = 0;\n    }\n}\n__kernel void replicate_8409(int32_t x_8197, __global unsigned char *mem_8360)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t replicate_gtid_8409;\n    int32_t replicate_ltid_8410;\n    int32_t replicate_gid_8411;\n    \n    replicate_gtid_8409 = get_global_id(0);\n    replicate_ltid_8410 = get_local_id(0);\n    replicate_gid_8411 = get_group_id(0);\n    if (slt32(replicate_gtid_8409, x_8197)) {\n        *(__global int64_t *) &mem_8360[replicate_gtid_8409 * 8] = -1;\n    }\n}\n__kernel void replicate_8414(__global unsigned char *mem_8363)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t replicate_gtid_8414;\n    int32_t replicate_ltid_8415;\n    int32_t replicate_gid_8416;\n    \n    replicate_gtid_8414 = get_global_id(0);\n    replicate_ltid_8415 = get_local_id(0);\n    replicate_gid_8416 = get_group_id(0);\n    if (slt32(replicate_gtid_8414, 256)) {\n        *(__global int64_t *) &mem_8363[replicate_gtid_8414 * 8] = 0;\n    }\n}\n",
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
static const char *size_names[] = {"_format.group_size_8291",
                                   "_myers.group_size_8303",
                                   "_myers.group_size_8402",
                                   "_myers.group_size_8407",
                                   "_myers.group_size_8412",
                                   "_myers.group_size_8417"};
static const char *size_vars[] = {"_formatzigroup_sizze_8291",
                                  "_myerszigroup_sizze_8303",
                                  "_myerszigroup_sizze_8402",
                                  "_myerszigroup_sizze_8407",
                                  "_myerszigroup_sizze_8412",
                                  "_myerszigroup_sizze_8417"};
static const char *size_classes[] = {"group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size"};
int futhark_get_num_sizes(void)
{
    return 6;
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
    size_t _formatzigroup_sizze_8291;
    size_t _myerszigroup_sizze_8303;
    size_t _myerszigroup_sizze_8402;
    size_t _myerszigroup_sizze_8407;
    size_t _myerszigroup_sizze_8412;
    size_t _myerszigroup_sizze_8417;
} ;
struct futhark_context_config {
    struct cuda_config cu_cfg;
    size_t sizes[6];
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
    cfg->sizes[5] = 0;
    cuda_config_init(&cfg->cu_cfg, 6, size_names, size_vars, cfg->sizes,
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
    for (int i = 0; i < 6; i++) {
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
    CUfunction map_8297;
    CUfunction map_8309;
    CUfunction replicate_8399;
    CUfunction replicate_8404;
    CUfunction replicate_8409;
    CUfunction replicate_8414;
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
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->map_8297, ctx->cuda.module,
                                     "map_8297"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->map_8309, ctx->cuda.module,
                                     "map_8309"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->replicate_8399, ctx->cuda.module,
                                     "replicate_8399"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->replicate_8404, ctx->cuda.module,
                                     "replicate_8404"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->replicate_8409, ctx->cuda.module,
                                     "replicate_8409"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->replicate_8414, ctx->cuda.module,
                                     "replicate_8414"));
    ctx->sizes._formatzigroup_sizze_8291 = cfg->sizes[0];
    ctx->sizes._myerszigroup_sizze_8303 = cfg->sizes[1];
    ctx->sizes._myerszigroup_sizze_8402 = cfg->sizes[2];
    ctx->sizes._myerszigroup_sizze_8407 = cfg->sizes[3];
    ctx->sizes._myerszigroup_sizze_8412 = cfg->sizes[4];
    ctx->sizes._myerszigroup_sizze_8417 = cfg->sizes[5];
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
static int futrts__myers(struct futhark_context *ctx,
                         int64_t *out_out_memsizze_8428,
                         struct memblock_device *out_mem_p_8429,
                         int32_t *out_out_arrsizze_8430,
                         int64_t x_mem_sizze_8348,
                         struct memblock_device x_mem_8349,
                         int64_t x_mem_sizze_8350,
                         struct memblock_device x_mem_8351, int32_t sizze_8193,
                         int32_t sizze_8194, int32_t x_8197);
static int futrts__format(struct futhark_context *ctx,
                          int64_t *out_out_memsizze_8461,
                          struct memblock_device *out_mem_p_8462,
                          int32_t *out_out_arrsizze_8463,
                          int64_t x_mem_sizze_8348,
                          struct memblock_device x_mem_8349, int32_t sizze_8184,
                          int32_t sizze_8185);
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
static int futrts__myers(struct futhark_context *ctx,
                         int64_t *out_out_memsizze_8428,
                         struct memblock_device *out_mem_p_8429,
                         int32_t *out_out_arrsizze_8430,
                         int64_t x_mem_sizze_8348,
                         struct memblock_device x_mem_8349,
                         int64_t x_mem_sizze_8350,
                         struct memblock_device x_mem_8351, int32_t sizze_8193,
                         int32_t sizze_8194, int32_t x_8197)
{
    int64_t out_memsizze_8397;
    struct memblock_device out_mem_8396;
    
    out_mem_8396.references = NULL;
    
    int32_t out_arrsizze_8398;
    int64_t arg_8198 = zext_i32_i64(sizze_8193);
    int64_t y_8199 = arg_8198 - 1;
    int64_t res_8200 = 1 << y_8199;
    int16_t arg_8201 = zext_i32_i16(sizze_8193);
    bool bounds_invalid_upwards_8202 = slt32(x_8197, 0);
    bool eq_x_zz_8203 = 0 == x_8197;
    bool not_p_8204 = !bounds_invalid_upwards_8202;
    bool p_and_eq_x_y_8205 = eq_x_zz_8203 && not_p_8204;
    bool dim_zzero_8206 = bounds_invalid_upwards_8202 || p_and_eq_x_y_8205;
    bool both_empty_8207 = eq_x_zz_8203 && dim_zzero_8206;
    bool empty_or_match_8211 = not_p_8204 || both_empty_8207;
    bool empty_or_match_cert_8212;
    
    if (!empty_or_match_8211) {
        ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%s\n",
                               "meyers-opti.fut:60:1-20 -> meyers-opti.fut:25:13-35 -> /futlib/array.fut:66:1-67:19",
                               "Function return value does not match shape of type ",
                               "*", "[", x_8197, "]", "t");
        if (memblock_unref_device(ctx, &out_mem_8396, "out_mem_8396") != 0)
            return 1;
        return 1;
    }
    
    int64_t binop_x_8353 = sext_i32_i64(x_8197);
    int64_t bytes_8352 = 2 * binop_x_8353;
    struct memblock_device mem_8354;
    
    mem_8354.references = NULL;
    if (memblock_alloc_device(ctx, &mem_8354, bytes_8352, "mem_8354"))
        return 1;
    
    int32_t group_sizze_8402;
    
    group_sizze_8402 = ctx->sizes._myerszigroup_sizze_8402;
    
    int32_t num_groups_8403;
    
    num_groups_8403 = squot32(x_8197 + sext_i32_i32(group_sizze_8402) - 1,
                              sext_i32_i32(group_sizze_8402));
    
    CUdeviceptr kernel_arg_8434 = mem_8354.mem;
    
    if ((((((1 && num_groups_8403 != 0) && 1 != 0) && 1 != 0) &&
          group_sizze_8402 != 0) && 1 != 0) && 1 != 0) {
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
        
        grid[perm[0]] = num_groups_8403;
        grid[perm[1]] = 1;
        grid[perm[2]] = 1;
        
        void *kernel_args_8431[] = {&x_8197, &arg_8201, &kernel_arg_8434};
        int64_t time_start_8432 = 0, time_end_8433 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with grid size (", "replicate_8399");
            fprintf(stderr, "%d", num_groups_8403);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ") and block size (");
            fprintf(stderr, "%d", group_sizze_8402);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ").\n");
            time_start_8432 = get_wall_time();
        }
        CUDA_SUCCEED(cuLaunchKernel(ctx->replicate_8399, grid[0], grid[1],
                                    grid[2], group_sizze_8402, 1, 1, 0, NULL,
                                    kernel_args_8431, NULL));
        if (ctx->debugging) {
            CUDA_SUCCEED(cuCtxSynchronize());
            time_end_8433 = get_wall_time();
            fprintf(stderr, "Kernel %s runtime: %ldus\n", "replicate_8399",
                    time_end_8433 - time_start_8432);
        }
    }
    
    int64_t bytes_8355 = 8 * binop_x_8353;
    struct memblock_device mem_8357;
    
    mem_8357.references = NULL;
    if (memblock_alloc_device(ctx, &mem_8357, bytes_8355, "mem_8357"))
        return 1;
    
    int32_t group_sizze_8407;
    
    group_sizze_8407 = ctx->sizes._myerszigroup_sizze_8407;
    
    int32_t num_groups_8408;
    
    num_groups_8408 = squot32(x_8197 + sext_i32_i32(group_sizze_8407) - 1,
                              sext_i32_i32(group_sizze_8407));
    
    CUdeviceptr kernel_arg_8438 = mem_8357.mem;
    
    if ((((((1 && num_groups_8408 != 0) && 1 != 0) && 1 != 0) &&
          group_sizze_8407 != 0) && 1 != 0) && 1 != 0) {
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
        
        grid[perm[0]] = num_groups_8408;
        grid[perm[1]] = 1;
        grid[perm[2]] = 1;
        
        void *kernel_args_8435[] = {&x_8197, &kernel_arg_8438};
        int64_t time_start_8436 = 0, time_end_8437 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with grid size (", "replicate_8404");
            fprintf(stderr, "%d", num_groups_8408);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ") and block size (");
            fprintf(stderr, "%d", group_sizze_8407);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ").\n");
            time_start_8436 = get_wall_time();
        }
        CUDA_SUCCEED(cuLaunchKernel(ctx->replicate_8404, grid[0], grid[1],
                                    grid[2], group_sizze_8407, 1, 1, 0, NULL,
                                    kernel_args_8435, NULL));
        if (ctx->debugging) {
            CUDA_SUCCEED(cuCtxSynchronize());
            time_end_8437 = get_wall_time();
            fprintf(stderr, "Kernel %s runtime: %ldus\n", "replicate_8404",
                    time_end_8437 - time_start_8436);
        }
    }
    
    struct memblock_device mem_8360;
    
    mem_8360.references = NULL;
    if (memblock_alloc_device(ctx, &mem_8360, bytes_8355, "mem_8360"))
        return 1;
    
    int32_t group_sizze_8412;
    
    group_sizze_8412 = ctx->sizes._myerszigroup_sizze_8412;
    
    int32_t num_groups_8413;
    
    num_groups_8413 = squot32(x_8197 + sext_i32_i32(group_sizze_8412) - 1,
                              sext_i32_i32(group_sizze_8412));
    
    CUdeviceptr kernel_arg_8442 = mem_8360.mem;
    
    if ((((((1 && num_groups_8413 != 0) && 1 != 0) && 1 != 0) &&
          group_sizze_8412 != 0) && 1 != 0) && 1 != 0) {
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
        
        grid[perm[0]] = num_groups_8413;
        grid[perm[1]] = 1;
        grid[perm[2]] = 1;
        
        void *kernel_args_8439[] = {&x_8197, &kernel_arg_8442};
        int64_t time_start_8440 = 0, time_end_8441 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with grid size (", "replicate_8409");
            fprintf(stderr, "%d", num_groups_8413);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ") and block size (");
            fprintf(stderr, "%d", group_sizze_8412);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ").\n");
            time_start_8440 = get_wall_time();
        }
        CUDA_SUCCEED(cuLaunchKernel(ctx->replicate_8409, grid[0], grid[1],
                                    grid[2], group_sizze_8412, 1, 1, 0, NULL,
                                    kernel_args_8439, NULL));
        if (ctx->debugging) {
            CUDA_SUCCEED(cuCtxSynchronize());
            time_end_8441 = get_wall_time();
            fprintf(stderr, "Kernel %s runtime: %ldus\n", "replicate_8409",
                    time_end_8441 - time_start_8440);
        }
    }
    
    struct memblock_device mem_8363;
    
    mem_8363.references = NULL;
    if (memblock_alloc_device(ctx, &mem_8363, 2048, "mem_8363"))
        return 1;
    
    int32_t group_sizze_8417;
    
    group_sizze_8417 = ctx->sizes._myerszigroup_sizze_8417;
    
    int32_t num_groups_8418;
    
    num_groups_8418 = squot32(256 + sext_i32_i32(group_sizze_8417) - 1,
                              sext_i32_i32(group_sizze_8417));
    
    CUdeviceptr kernel_arg_8446 = mem_8363.mem;
    
    if ((((((1 && num_groups_8418 != 0) && 1 != 0) && 1 != 0) &&
          group_sizze_8417 != 0) && 1 != 0) && 1 != 0) {
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
        
        grid[perm[0]] = num_groups_8418;
        grid[perm[1]] = 1;
        grid[perm[2]] = 1;
        
        void *kernel_args_8443[] = {&kernel_arg_8446};
        int64_t time_start_8444 = 0, time_end_8445 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with grid size (", "replicate_8414");
            fprintf(stderr, "%d", num_groups_8418);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ") and block size (");
            fprintf(stderr, "%d", group_sizze_8417);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ").\n");
            time_start_8444 = get_wall_time();
        }
        CUDA_SUCCEED(cuLaunchKernel(ctx->replicate_8414, grid[0], grid[1],
                                    grid[2], group_sizze_8417, 1, 1, 0, NULL,
                                    kernel_args_8443, NULL));
        if (ctx->debugging) {
            CUDA_SUCCEED(cuCtxSynchronize());
            time_end_8445 = get_wall_time();
            fprintf(stderr, "Kernel %s runtime: %ldus\n", "replicate_8414",
                    time_end_8445 - time_start_8444);
        }
    }
    
    bool empty_or_match_cert_8217;
    
    if (!empty_or_match_8211) {
        ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%s\n",
                               "meyers-opti.fut:60:1-20 -> meyers-opti.fut:29:13-18 -> /futlib/array.fut:61:1-62:12",
                               "Function return value does not match shape of type ",
                               "*", "[", x_8197, "]", "intrinsics.i32");
        if (memblock_unref_device(ctx, &mem_8363, "mem_8363") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_8360, "mem_8360") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_8357, "mem_8357") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_8354, "mem_8354") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_8396, "out_mem_8396") != 0)
            return 1;
        return 1;
    }
    for (int32_t i_8221 = 0; i_8221 < sizze_8193; i_8221++) {
        int16_t read_res_8447;
        
        CUDA_SUCCEED(cuMemcpyDtoH(&read_res_8447, x_mem_8349.mem + i_8221 * 2,
                                  sizeof(int16_t)));
        
        int16_t arg_8226 = read_res_8447;
        int32_t res_8227 = zext_i16_i32(arg_8226);
        bool x_8228 = sle32(0, res_8227);
        bool y_8229 = slt32(res_8227, 256);
        bool bounds_check_8230 = x_8228 && y_8229;
        bool index_certs_8231;
        
        if (!bounds_check_8230) {
            ctx->error = msgprintf("Error at %s:\n%s%d%s%d%s\n",
                                   "meyers-opti.fut:60:1-20 -> meyers-opti.fut:33:22-26",
                                   "Index [", res_8227,
                                   "] out of bounds for array of shape [", 256,
                                   "].");
            if (memblock_unref_device(ctx, &mem_8363, "mem_8363") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_8360, "mem_8360") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_8357, "mem_8357") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_8354, "mem_8354") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_8396, "out_mem_8396") != 0)
                return 1;
            return 1;
        }
        
        int64_t read_res_8448;
        
        CUDA_SUCCEED(cuMemcpyDtoH(&read_res_8448, mem_8363.mem + res_8227 * 8,
                                  sizeof(int64_t)));
        
        int64_t x_8232 = read_res_8448;
        int64_t arg_8233 = zext_i32_i64(i_8221);
        int64_t y_8234 = 1 << arg_8233;
        int64_t lw_val_8235 = x_8232 | y_8234;
        int64_t write_tmp_8449 = lw_val_8235;
        
        CUDA_SUCCEED(cuMemcpyHtoD(mem_8363.mem + res_8227 * 8, &write_tmp_8449,
                                  sizeof(int64_t)));
    }
    
    int32_t s_sign_8237 = (x_8197 > 0) - (x_8197 < 0);
    bool bounds_invalid_upwards_8238 = slt32(sizze_8194, 0);
    bool step_wrong_dir_8239 = s_sign_8237 == -1;
    bool step_invalid_8240 = eq_x_zz_8203 || step_wrong_dir_8239;
    bool range_invalid_8241 = bounds_invalid_upwards_8238 || step_invalid_8240;
    int32_t pos_step_8242 = x_8197 * s_sign_8237;
    int32_t num_elems_8243;
    
    if (range_invalid_8241) {
        num_elems_8243 = 0;
    } else {
        int32_t y_8244 = pos_step_8242 - 1;
        int32_t x_8245 = sizze_8194 + y_8244;
        int32_t x_8246 = squot32(x_8245, pos_step_8242);
        
        num_elems_8243 = x_8246;
    }
    
    bool loop_nonempty_8345 = slt32(0, num_elems_8243);
    int32_t group_sizze_8304;
    
    group_sizze_8304 = ctx->sizes._myerszigroup_sizze_8303;
    
    int32_t y_8305 = group_sizze_8304 - 1;
    int32_t x_8306 = x_8197 + y_8305;
    int32_t num_groups_8307;
    
    if (loop_nonempty_8345) {
        int32_t x_8346 = squot32(x_8306, group_sizze_8304);
        
        num_groups_8307 = x_8346;
    } else {
        num_groups_8307 = 0;
    }
    
    int32_t num_threads_8308 = group_sizze_8304 * num_groups_8307;
    struct memblock_device res_mem_8384;
    
    res_mem_8384.references = NULL;
    
    struct memblock_device res_mem_8386;
    
    res_mem_8386.references = NULL;
    
    struct memblock_device res_mem_8388;
    
    res_mem_8388.references = NULL;
    
    struct memblock_device scs_mem_8369;
    
    scs_mem_8369.references = NULL;
    
    struct memblock_device mvs_mem_8371;
    
    mvs_mem_8371.references = NULL;
    
    struct memblock_device pvs_mem_8373;
    
    pvs_mem_8373.references = NULL;
    if (memblock_set_device(ctx, &scs_mem_8369, &mem_8354, "mem_8354") != 0)
        return 1;
    if (memblock_set_device(ctx, &mvs_mem_8371, &mem_8357, "mem_8357") != 0)
        return 1;
    if (memblock_set_device(ctx, &pvs_mem_8373, &mem_8360, "mem_8360") != 0)
        return 1;
    for (int32_t i_8253 = 0; i_8253 < num_elems_8243; i_8253++) {
        int32_t index_primexp_8254 = x_8197 * i_8253;
        struct memblock_device mem_8376;
        
        mem_8376.references = NULL;
        if (memblock_alloc_device(ctx, &mem_8376, bytes_8355, "mem_8376"))
            return 1;
        
        struct memblock_device mem_8379;
        
        mem_8379.references = NULL;
        if (memblock_alloc_device(ctx, &mem_8379, bytes_8355, "mem_8379"))
            return 1;
        
        struct memblock_device mem_8382;
        
        mem_8382.references = NULL;
        if (memblock_alloc_device(ctx, &mem_8382, bytes_8352, "mem_8382"))
            return 1;
        
        CUdeviceptr kernel_arg_8453 = x_mem_8351.mem;
        CUdeviceptr kernel_arg_8454 = mem_8363.mem;
        CUdeviceptr kernel_arg_8455 = scs_mem_8369.mem;
        CUdeviceptr kernel_arg_8456 = mvs_mem_8371.mem;
        CUdeviceptr kernel_arg_8457 = pvs_mem_8373.mem;
        CUdeviceptr kernel_arg_8458 = mem_8376.mem;
        CUdeviceptr kernel_arg_8459 = mem_8379.mem;
        CUdeviceptr kernel_arg_8460 = mem_8382.mem;
        
        if ((((((1 && num_groups_8307 != 0) && 1 != 0) && 1 != 0) &&
              group_sizze_8304 != 0) && 1 != 0) && 1 != 0) {
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
            
            grid[perm[0]] = num_groups_8307;
            grid[perm[1]] = 1;
            grid[perm[2]] = 1;
            
            void *kernel_args_8450[] = {&x_8197, &res_8200, &index_primexp_8254,
                                        &kernel_arg_8453, &kernel_arg_8454,
                                        &kernel_arg_8455, &kernel_arg_8456,
                                        &kernel_arg_8457, &kernel_arg_8458,
                                        &kernel_arg_8459, &kernel_arg_8460};
            int64_t time_start_8451 = 0, time_end_8452 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with grid size (", "map_8309");
                fprintf(stderr, "%d", num_groups_8307);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ") and block size (");
                fprintf(stderr, "%d", group_sizze_8304);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ").\n");
                time_start_8451 = get_wall_time();
            }
            CUDA_SUCCEED(cuLaunchKernel(ctx->map_8309, grid[0], grid[1],
                                        grid[2], group_sizze_8304, 1, 1, 0,
                                        NULL, kernel_args_8450, NULL));
            if (ctx->debugging) {
                CUDA_SUCCEED(cuCtxSynchronize());
                time_end_8452 = get_wall_time();
                fprintf(stderr, "Kernel %s runtime: %ldus\n", "map_8309",
                        time_end_8452 - time_start_8451);
            }
        }
        
        struct memblock_device scs_mem_tmp_8420;
        
        scs_mem_tmp_8420.references = NULL;
        if (memblock_set_device(ctx, &scs_mem_tmp_8420, &mem_8382,
                                "mem_8382") != 0)
            return 1;
        
        struct memblock_device mvs_mem_tmp_8421;
        
        mvs_mem_tmp_8421.references = NULL;
        if (memblock_set_device(ctx, &mvs_mem_tmp_8421, &mem_8376,
                                "mem_8376") != 0)
            return 1;
        
        struct memblock_device pvs_mem_tmp_8422;
        
        pvs_mem_tmp_8422.references = NULL;
        if (memblock_set_device(ctx, &pvs_mem_tmp_8422, &mem_8379,
                                "mem_8379") != 0)
            return 1;
        if (memblock_set_device(ctx, &scs_mem_8369, &scs_mem_tmp_8420,
                                "scs_mem_tmp_8420") != 0)
            return 1;
        if (memblock_set_device(ctx, &mvs_mem_8371, &mvs_mem_tmp_8421,
                                "mvs_mem_tmp_8421") != 0)
            return 1;
        if (memblock_set_device(ctx, &pvs_mem_8373, &pvs_mem_tmp_8422,
                                "pvs_mem_tmp_8422") != 0)
            return 1;
        if (memblock_unref_device(ctx, &pvs_mem_tmp_8422, "pvs_mem_tmp_8422") !=
            0)
            return 1;
        if (memblock_unref_device(ctx, &mvs_mem_tmp_8421, "mvs_mem_tmp_8421") !=
            0)
            return 1;
        if (memblock_unref_device(ctx, &scs_mem_tmp_8420, "scs_mem_tmp_8420") !=
            0)
            return 1;
        if (memblock_unref_device(ctx, &mem_8382, "mem_8382") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_8379, "mem_8379") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_8376, "mem_8376") != 0)
            return 1;
    }
    if (memblock_set_device(ctx, &res_mem_8384, &scs_mem_8369,
                            "scs_mem_8369") != 0)
        return 1;
    if (memblock_set_device(ctx, &res_mem_8386, &mvs_mem_8371,
                            "mvs_mem_8371") != 0)
        return 1;
    if (memblock_set_device(ctx, &res_mem_8388, &pvs_mem_8373,
                            "pvs_mem_8373") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_8354, "mem_8354") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_8357, "mem_8357") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_8360, "mem_8360") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_8363, "mem_8363") != 0)
        return 1;
    out_arrsizze_8398 = x_8197;
    out_memsizze_8397 = bytes_8352;
    if (memblock_set_device(ctx, &out_mem_8396, &res_mem_8384,
                            "res_mem_8384") != 0)
        return 1;
    *out_out_memsizze_8428 = out_memsizze_8397;
    (*out_mem_p_8429).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_8429, &out_mem_8396,
                            "out_mem_8396") != 0)
        return 1;
    *out_out_arrsizze_8430 = out_arrsizze_8398;
    if (memblock_unref_device(ctx, &pvs_mem_8373, "pvs_mem_8373") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mvs_mem_8371, "mvs_mem_8371") != 0)
        return 1;
    if (memblock_unref_device(ctx, &scs_mem_8369, "scs_mem_8369") != 0)
        return 1;
    if (memblock_unref_device(ctx, &res_mem_8388, "res_mem_8388") != 0)
        return 1;
    if (memblock_unref_device(ctx, &res_mem_8386, "res_mem_8386") != 0)
        return 1;
    if (memblock_unref_device(ctx, &res_mem_8384, "res_mem_8384") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_8363, "mem_8363") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_8360, "mem_8360") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_8357, "mem_8357") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_8354, "mem_8354") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_8396, "out_mem_8396") != 0)
        return 1;
    return 0;
}
static int futrts__format(struct futhark_context *ctx,
                          int64_t *out_out_memsizze_8461,
                          struct memblock_device *out_mem_p_8462,
                          int32_t *out_out_arrsizze_8463,
                          int64_t x_mem_sizze_8348,
                          struct memblock_device x_mem_8349, int32_t sizze_8184,
                          int32_t sizze_8185)
{
    int64_t out_memsizze_8392;
    struct memblock_device out_mem_8391;
    
    out_mem_8391.references = NULL;
    
    int32_t out_arrsizze_8393;
    int32_t flat_dim_8188 = sizze_8184 * sizze_8185;
    int32_t group_sizze_8292;
    
    group_sizze_8292 = ctx->sizes._formatzigroup_sizze_8291;
    
    int32_t y_8293 = group_sizze_8292 - 1;
    int32_t x_8294 = flat_dim_8188 + y_8293;
    int32_t num_groups_8295 = squot32(x_8294, group_sizze_8292);
    int32_t num_threads_8296 = group_sizze_8292 * num_groups_8295;
    int64_t binop_x_8351 = sext_i32_i64(flat_dim_8188);
    int64_t bytes_8350 = 2 * binop_x_8351;
    struct memblock_device mem_8352;
    
    mem_8352.references = NULL;
    if (memblock_alloc_device(ctx, &mem_8352, bytes_8350, "mem_8352"))
        return 1;
    
    CUdeviceptr kernel_arg_8467 = x_mem_8349.mem;
    CUdeviceptr kernel_arg_8468 = mem_8352.mem;
    
    if ((((((1 && num_groups_8295 != 0) && 1 != 0) && 1 != 0) &&
          group_sizze_8292 != 0) && 1 != 0) && 1 != 0) {
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
        
        grid[perm[0]] = num_groups_8295;
        grid[perm[1]] = 1;
        grid[perm[2]] = 1;
        
        void *kernel_args_8464[] = {&sizze_8184, &sizze_8185, &flat_dim_8188,
                                    &kernel_arg_8467, &kernel_arg_8468};
        int64_t time_start_8465 = 0, time_end_8466 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with grid size (", "map_8297");
            fprintf(stderr, "%d", num_groups_8295);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ") and block size (");
            fprintf(stderr, "%d", group_sizze_8292);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ").\n");
            time_start_8465 = get_wall_time();
        }
        CUDA_SUCCEED(cuLaunchKernel(ctx->map_8297, grid[0], grid[1], grid[2],
                                    group_sizze_8292, 1, 1, 0, NULL,
                                    kernel_args_8464, NULL));
        if (ctx->debugging) {
            CUDA_SUCCEED(cuCtxSynchronize());
            time_end_8466 = get_wall_time();
            fprintf(stderr, "Kernel %s runtime: %ldus\n", "map_8297",
                    time_end_8466 - time_start_8465);
        }
    }
    out_arrsizze_8393 = flat_dim_8188;
    out_memsizze_8392 = bytes_8350;
    if (memblock_set_device(ctx, &out_mem_8391, &mem_8352, "mem_8352") != 0)
        return 1;
    *out_out_memsizze_8461 = out_memsizze_8392;
    (*out_mem_p_8462).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_8462, &out_mem_8391,
                            "out_mem_8391") != 0)
        return 1;
    *out_out_arrsizze_8463 = out_arrsizze_8393;
    if (memblock_unref_device(ctx, &mem_8352, "mem_8352") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_8391, "out_mem_8391") != 0)
        return 1;
    return 0;
}
struct futhark_i32_2d {
    struct memblock_device mem;
    int64_t shape[2];
} ;
struct futhark_i32_2d *futhark_new_i32_2d(struct futhark_context *ctx,
                                          int32_t *data, int dim0, int dim1)
{
    struct futhark_i32_2d *bad = NULL;
    struct futhark_i32_2d *arr = malloc(sizeof(struct futhark_i32_2d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, dim0 * dim1 * sizeof(int32_t),
                              "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    CUDA_SUCCEED(cuMemcpyHtoD(arr->mem.mem + 0, data + 0, dim0 * dim1 *
                              sizeof(int32_t)));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_i32_2d *futhark_new_raw_i32_2d(struct futhark_context *ctx,
                                              CUdeviceptr data, int offset,
                                              int dim0, int dim1)
{
    struct futhark_i32_2d *bad = NULL;
    struct futhark_i32_2d *arr = malloc(sizeof(struct futhark_i32_2d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, dim0 * dim1 * sizeof(int32_t),
                              "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    CUDA_SUCCEED(cuMemcpy(arr->mem.mem + 0, data + offset, dim0 * dim1 *
                          sizeof(int32_t)));
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_i32_2d(struct futhark_context *ctx, struct futhark_i32_2d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref_device(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_i32_2d(struct futhark_context *ctx,
                          struct futhark_i32_2d *arr, int32_t *data)
{
    lock_lock(&ctx->lock);
    CUDA_SUCCEED(cuMemcpyDtoH(data + 0, arr->mem.mem + 0, arr->shape[0] *
                              arr->shape[1] * sizeof(int32_t)));
    lock_unlock(&ctx->lock);
    return 0;
}
CUdeviceptr futhark_values_raw_i32_2d(struct futhark_context *ctx,
                                      struct futhark_i32_2d *arr)
{
    return arr->mem.mem;
}
int64_t *futhark_shape_i32_2d(struct futhark_context *ctx,
                              struct futhark_i32_2d *arr)
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
int futhark_entry__myers(struct futhark_context *ctx,
                         struct futhark_u16_1d **out0, const
                         struct futhark_u16_1d *in0, const
                         struct futhark_u16_1d *in1, const int32_t in2)
{
    int64_t x_mem_sizze_8348;
    struct memblock_device x_mem_8349;
    
    x_mem_8349.references = NULL;
    
    int64_t x_mem_sizze_8350;
    struct memblock_device x_mem_8351;
    
    x_mem_8351.references = NULL;
    
    int32_t sizze_8193;
    int32_t sizze_8194;
    int32_t x_8197;
    int64_t out_memsizze_8397;
    struct memblock_device out_mem_8396;
    
    out_mem_8396.references = NULL;
    
    int32_t out_arrsizze_8398;
    
    lock_lock(&ctx->lock);
    x_mem_8349 = in0->mem;
    x_mem_sizze_8348 = in0->mem.size;
    sizze_8193 = in0->shape[0];
    x_mem_8351 = in1->mem;
    x_mem_sizze_8350 = in1->mem.size;
    sizze_8194 = in1->shape[0];
    x_8197 = in2;
    
    int ret = futrts__myers(ctx, &out_memsizze_8397, &out_mem_8396,
                            &out_arrsizze_8398, x_mem_sizze_8348, x_mem_8349,
                            x_mem_sizze_8350, x_mem_8351, sizze_8193,
                            sizze_8194, x_8197);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_u16_1d))) != NULL);
        (*out0)->mem = out_mem_8396;
        (*out0)->shape[0] = out_arrsizze_8398;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry__format(struct futhark_context *ctx,
                          struct futhark_u16_1d **out0, const
                          struct futhark_i32_2d *in0)
{
    int64_t x_mem_sizze_8348;
    struct memblock_device x_mem_8349;
    
    x_mem_8349.references = NULL;
    
    int32_t sizze_8184;
    int32_t sizze_8185;
    int64_t out_memsizze_8392;
    struct memblock_device out_mem_8391;
    
    out_mem_8391.references = NULL;
    
    int32_t out_arrsizze_8393;
    
    lock_lock(&ctx->lock);
    x_mem_8349 = in0->mem;
    x_mem_sizze_8348 = in0->mem.size;
    sizze_8184 = in0->shape[0];
    sizze_8185 = in0->shape[1];
    
    int ret = futrts__format(ctx, &out_memsizze_8392, &out_mem_8391,
                             &out_arrsizze_8393, x_mem_sizze_8348, x_mem_8349,
                             sizze_8184, sizze_8185);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_u16_1d))) != NULL);
        (*out0)->mem = out_mem_8391;
        (*out0)->shape[0] = out_arrsizze_8393;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
