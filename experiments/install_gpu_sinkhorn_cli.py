#!/usr/bin/env python3
"""Apply guarded GPU Sinkhorn CLI edits on the user's machine; run no builds."""

from pathlib import Path
import os
import re
import tempfile
import textwrap


ROOT = Path(__file__).resolve().parents[1]
MARKER = "// GPU_SINKHORN_CLI_V1"


def once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one edit anchor: {old[:100]!r}; no files saved")
    return text.replace(old, new, 1)


def main_update(text):
    if MARKER in text:
        return text
    if '#include "gpu_sinkhorn.hpp"' not in text:
        text = once(text, '#include "gpu_assignment.hpp"',
                    '#include "gpu_assignment.hpp"\n#include "gpu_sinkhorn.hpp"')
    if '#include <optional>' not in text:
        text = once(text, '#include "gpu_sinkhorn.hpp"',
                    '#include "gpu_sinkhorn.hpp"\n#include <optional>')
    text = once(text, '    std::string assignment_backend = "cpu";',
                '    std::string assignment_backend = "cpu";\n'
                '    std::string sinkhorn_backend = "cpu";\n'
                '    std::string sinkhorn_output;')
    text = once(text, '    const bool indexed = options.backend == "structural-index";',
                '    const bool indexed = options.backend == "structural-index";\n'
                '    if (options.sinkhorn_backend == "gpu" && N != M) {\n'
                '        throw std::invalid_argument("GPU Sinkhorn requires a square matrix");\n'
                '    }')
    declarations = '''
    constexpr double sinkhorn_epsilon = 0.1;
    constexpr int sinkhorn_max_iterations = 500;
    constexpr double sinkhorn_tolerance = 1e-3;
    SparseSinkhornResult sinkhorn_result;
    GPUSinkhornDiagnostics gpu_sinkhorn_diagnostics;
    std::optional<CandidateSupportStats> cached_support;
    bool gpu_sinkhorn_executed = false;
    bool cpu_sinkhorn_executed = false;
    bool sinkhorn_confidence_exported = false;
    std::size_t sinkhorn_download_bytes = 0;
    double sinkhorn_ms = 0.0;
    std::string sinkhorn_status = "not_run";
'''
    # Replace the original CPU Sinkhorn block before moving its declarations.
    start = text.index('    constexpr double sinkhorn_epsilon = 0.1;')
    end = text.index('    device_memory_sampler.stop();', start)
    old_block = text[start:end]
    if old_block.count('sparse_sinkhorn(') != 1 or \
            'SparseSinkhornResult sinkhorn_result;' not in old_block:
        raise RuntimeError("Unexpected existing Sinkhorn block; no files saved")
    final_stage = r'''
    if (assignment.feasible && is_one_to_one(assignment.row_to_col, M)) {
        if (options.sinkhorn_backend == "gpu") {
            if (!gpu_sinkhorn_executed) {
                throw std::runtime_error("GPU Sinkhorn was requested but did not execute");
            }
        } else {
            Timer timer("CPU sparse Sinkhorn");
            sinkhorn_result = sparse_sinkhorn(
                csr, sinkhorn_epsilon, sinkhorn_max_iterations, sinkhorn_tolerance);
            sinkhorn_ms = timer.elapsed_ms();
            cpu_sinkhorn_executed = true;
            sinkhorn_status = sinkhorn_result.converged ? "converged" : "iteration_limit";
        }
        std::cout << "Sparse Sinkhorn (" << options.sinkhorn_backend << "): "
                  << sinkhorn_status << " after " << sinkhorn_result.iterations
                  << " iterations; max row error=" << sinkhorn_result.max_row_marginal_error
                  << ", max column error=" << sinkhorn_result.max_column_marginal_error
                  << std::endl;
    } else if (!gpu_sinkhorn_executed) {
        sinkhorn_status = "skipped_infeasible_assignment";
    }

    // Confidence output never modifies the hard assignment.
    if (!options.sinkhorn_output.empty() &&
        (cpu_sinkhorn_executed ||
         (gpu_sinkhorn_executed && gpu_sinkhorn_diagnostics.probabilities_valid))) {
        if (sinkhorn_result.edge_probabilities.size() != csr.values.size()) {
            throw std::runtime_error("Sinkhorn confidence count disagrees with CSR");
        }
        std::ofstream confidence_out(options.sinkhorn_output);
        if (!confidence_out) {
            throw std::runtime_error("Cannot open Sinkhorn output: " + options.sinkhorn_output);
        }
        confidence_out << std::setprecision(std::numeric_limits<double>::max_digits10);
        confidence_out << "source,target,probability\n";
        for (int row = 0; row < csr.rows; ++row) {
            for (int p = csr.row_ptr[row]; p < csr.row_ptr[row + 1]; ++p) {
                confidence_out << row << ',' << csr.col_idx[p] << ','
                    << sinkhorn_result.edge_probabilities[p] << '\n';
            }
        }
        confidence_out.close();
        if (!confidence_out) throw std::runtime_error("Failed to write Sinkhorn output");
        sinkhorn_confidence_exported = true;
    }

    std::string pipeline_description = !candidate_input_file.empty()
        ? "Fixed candidate support/costs"
        : (use_gpu ? "GPU feature/candidate processing" : "CPU feature/candidate processing");
    pipeline_description += "; assignment requested: " + options.assignment_backend;
    pipeline_description += "; Sinkhorn requested: " + options.sinkhorn_backend;

'''
    text = text[:start] + final_stage.lstrip('\n') + text[end:]
    text = once(text, '    bool gpu_assignment_ready = false;\n',
                '    bool gpu_assignment_ready = false;\n' + declarations.lstrip('\n'))
    early_gpu = '''
if (options.sinkhorn_backend == "gpu") {
    cached_support = analyze_candidate_support(candidates, M);
    if (cached_support->maximum_matching_size == N) {
        Timer phase("GPU Sinkhorn and requested confidence handoff");
        {
            GPUSinkhornOptions parameters;
            parameters.epsilon = sinkhorn_epsilon;
            parameters.max_iterations = sinkhorn_max_iterations;
            parameters.tolerance = sinkhorn_tolerance;
            auto confidence = gpu_sparse_sinkhorn(device.costs, gpus[0], parameters);
            gpu_sinkhorn_diagnostics = confidence.diagnostics;
            sinkhorn_result.iterations = confidence.diagnostics.iterations;
            sinkhorn_result.max_row_marginal_error =
                confidence.diagnostics.max_row_marginal_error;
            sinkhorn_result.max_column_marginal_error =
                confidence.diagnostics.max_column_marginal_error;
            sinkhorn_result.converged = confidence.diagnostics.status ==
                GPUSinkhornStatus::Converged;
            sinkhorn_status = gpu_sinkhorn_status_name(confidence.diagnostics.status);
            if (!options.sinkhorn_output.empty() && confidence.diagnostics.probabilities_valid) {
                sinkhorn_result = gpu_download_sinkhorn(confidence, gpus[0]);
                sinkhorn_download_bytes = confidence.edge_probabilities.bytes();
            }
        }
        // Owned confidence/scaling buffers are gone before context cleanup.
        sinkhorn_ms = phase.elapsed_ms();
        gpu_sinkhorn_executed = true;
    }
}

'''
    match = re.search(r'(?m)^([ \t]+)gpu_selection = device\.selection;', text)
    if match is None:
        raise RuntimeError("Resident pipeline handoff not found; no files saved")
    text = text[:match.start()] + textwrap.indent(
        textwrap.dedent(early_gpu).lstrip('\n'), match[1]) + text[match.start():]
    text = once(text,
        '    const CandidateSupportStats support = analyze_candidate_support(candidates, M);',
        '    const CandidateSupportStats support = cached_support\n'
        '        ? std::move(*cached_support) : analyze_candidate_support(candidates, M);')
    unavailable = re.compile(
        r'if \(options\.assignment_backend == "gpu"\) \{\s*'
        r'throw std::runtime_error\(\s*'
        r'"GPU assignment was requested, but no GPU is available"\);\s*\}')
    text, count = unavailable.subn(
        'if (options.assignment_backend == "gpu" || options.sinkhorn_backend == "gpu") {\n'
        '                throw std::runtime_error(\n'
        '                    "A GPU stage was requested, but no GPU is available");\n'
        '            }', text)
    if count != 1:
        raise RuntimeError("GPU-unavailable branch differs; no files saved")

    text = once(text, '        if (arg == "--assignment") {', '''        if (arg == "--sinkhorn" || arg == "--sinkhorn-output") {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + arg);
            const std::string value = argv[++i];
            if (arg == "--sinkhorn") {
                if (value != "cpu" && value != "gpu") {
                    throw std::invalid_argument("--sinkhorn must be cpu or gpu");
                }
                options.sinkhorn_backend = value;
            } else {
                if (value.empty()) throw std::invalid_argument("Empty Sinkhorn output path");
                options.sinkhorn_output = value;
            }
        } else if (arg == "--assignment") {''')
    text = once(text, '            std::cout << "Usage: scgm_cuda [OPTIONS]" << std::endl;', r'''            std::cout << "  --sinkhorn cpu|gpu    Confidence backend (default cpu)\n"
                      << "  --sinkhorn-output FILE  Export edge confidence CSV\n";
            std::cout << "Usage: scgm_cuda [OPTIONS]" << std::endl;''')
    validation = '''    if (options.sinkhorn_backend == "gpu" &&
        (mode != "file" || !use_gpu || !options.resident_pipeline ||
         !candidate_input_file.empty())) {
        throw std::invalid_argument(
            "--sinkhorn gpu requires GPU file generation with --gpu-resident");
    }
    if (!options.sinkhorn_output.empty() && mode != "file") {
        throw std::invalid_argument("--sinkhorn-output requires file mode");
    }
'''
    text = once(text, '    if (options.assignment_backend == "gpu" &&\n',
                validation + '    if (options.assignment_backend == "gpu" &&\n')
    json_fields = r'''
        timings_out << "  \"requested_sinkhorn_backend\": \"" << options.sinkhorn_backend
                    << "\",\n  \"sinkhorn_backend\": \""
                    << (gpu_sinkhorn_executed ? "gpu" : (cpu_sinkhorn_executed ? "cpu" : "none"))
                    << "\",\n  \"sinkhorn_status\": \"" << sinkhorn_status
                    << "\",\n  \"sinkhorn_confidence_exported\": "
                    << (sinkhorn_confidence_exported ? "true" : "false") << ",\n";
        if (gpu_sinkhorn_executed) {
            const auto& s = gpu_sinkhorn_diagnostics;
            timings_out << "  \"gpu_sinkhorn\": {\"status\":\""
                << gpu_sinkhorn_status_name(s.status)
                << "\",\"probabilities_valid\":" << (s.probabilities_valid ? "true" : "false")
                << ",\"setup_ms\":" << s.setup_ms
                << ",\"reverse_index_build_ms\":" << s.reverse_index_build_ms
                << ",\"iteration_gpu_ms\":" << s.iteration_gpu_ms
                << ",\"solver_total_ms\":" << s.total_ms
                << ",\"workspace_bytes\":" << s.workspace_bytes
                << ",\"workspace_budget_bytes\":" << GPUSinkhornOptions{}.workspace_budget_bytes
                << ",\"persistent_result_bytes\":" << s.persistent_result_bytes
                << ",\"reverse_index_bytes\":" << s.reverse_index_bytes
                << ",\"confidence_download_bytes\":" << sinkhorn_download_bytes
                << ",\"marginals\":\"unit_rows_and_columns\"},\n";
        } else {
            timings_out << "  \"gpu_sinkhorn\": null,\n";
        }
'''
    text = once(text, '        timings_out << "{\\n";\n',
                '        timings_out << "{\\n";\n' + json_fields.lstrip('\n'))
    for field in ('max_row_marginal_error', 'max_column_marginal_error'):
        old = f'<< sinkhorn_result.{field} << ",\\n"'
        new = (f'<< measured_time_json(sinkhorn_result.{field}, '
               f'std::isfinite(sinkhorn_result.{field})) << ",\\n"')
        text = once(text, old, new)
    lines = text.splitlines(keepends=True)
    description = [i for i, line in enumerate(lines) if '"pipeline_description' in line or
                   '\\"pipeline_description' in line]
    if len(description) != 1:
        raise RuntimeError("Pipeline JSON description differs; no files saved")
    lines[description[0]] = r'''                    << "  \"pipeline_description\": \"" << pipeline_description << "\",\n"''' + '\n'
    text = ''.join(lines)
    text = once(text, '    return 0;\n}\n\nint main(int argc, char** argv)',
        '    // Preserve the hard mapping, but report a failed requested confidence stage.\n'
        '    if (gpu_sinkhorn_executed &&\n'
        '        gpu_sinkhorn_diagnostics.status != GPUSinkhornStatus::Converged) return 4;\n'
        '    return 0;\n}\n\nint main(int argc, char** argv)')
    return MARKER + '\n' + text


def cmake_update(text):
    # Normalize source registration, including any interrupted earlier edits.
    lines = text.splitlines(keepends=True)
    seen = False
    output = []
    for line in lines:
        if line.strip() == 'cuda/sparse_sinkhorn.cu':
            if seen:
                continue
            seen = True
        output.append(line)
    text = ''.join(output)
    if not seen:
        text = once(text, '    cuda/csr_matrix.cu\n',
                    '    cuda/csr_matrix.cu\n    cuda/sparse_sinkhorn.cu\n')
    block = '''    add_executable(test_gpu_sinkhorn tests/test_gpu_sinkhorn.cpp)
    target_link_libraries(test_gpu_sinkhorn PRIVATE scgm_lib)
    add_test(NAME test_gpu_sinkhorn COMMAND test_gpu_sinkhorn)
    set_tests_properties(test_gpu_sinkhorn PROPERTIES SKIP_RETURN_CODE 77)
'''
    text = text.replace(block, '')
    if 'test_gpu_sinkhorn' in text:
        raise RuntimeError("Unexpected Sinkhorn test registration; no files saved")
    return once(text, '    enable_testing()\n',
                '    enable_testing()\n\n' + block)


def main():
    paths = [ROOT / 'apps/main.cpp', ROOT / 'CMakeLists.txt']
    original = [path.read_text() for path in paths]
    updated = [main_update(original[0]), cmake_update(original[1])]
    # Validate all anchors before modifying either file.
    for path, before in zip(paths, original):
        if path.read_text() != before:
            raise RuntimeError(f"{path} changed concurrently; no files saved")
    for path, before, after in zip(paths, original, updated):
        if before == after:
            continue
        with tempfile.NamedTemporaryFile(mode='w', prefix='scgm-sinkhorn-backup-',
                                         suffix=path.suffix, delete=False) as backup:
            backup.write(before)
            print(f"Backup for {path.name}: {backup.name}")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=path.parent,
                    prefix='.sinkhorn-edit-', delete=False) as staged:
                staged.write(after)
                temporary = Path(staged.name)
            os.chmod(temporary, path.stat().st_mode & 0o777)
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    print('GPU Sinkhorn CLI and CMake registration are ready.')
    print('No builds, tests, or GPU workloads were run.')


if __name__ == '__main__':
    main()
