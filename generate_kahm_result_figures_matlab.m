function generate_kahm_result_figures_matlab(reportPath, outDir)
%GENERATE_KAHM_RESULT_FIGURES_MATLAB Create publication-ready figures from
% kahm_evaluation_report.md.
%
% This function reads the Markdown report produced by
% evaluate_three_embeddings_storylines.py and generates four figures:
%   1) quality_vs_k.eps
%   2) consensus_vs_k.eps
%   3) compute_quality_tradeoff.eps
%   5) figures_matlab_latex.tex
%
% Usage:
%   generate_kahm_result_figures_matlab
%   generate_kahm_result_figures_matlab('kahm_evaluation_report.md')
%   generate_kahm_result_figures_matlab('kahm_evaluation_report.md', 'kahm_figures_matlab')
%
% Notes:
% - The parser is tailored to the current report structure.
% - Requires a MATLAB version that supports local functions in scripts/functions.
% - exportgraphics is used when available; otherwise print is used as fallback.

    if nargin < 1 || isempty(reportPath)
        reportPath = 'kahm_evaluation_report.md';
    end
    if nargin < 2 || isempty(outDir)
        outDir = 'kahm_figures_matlab';
    end

    if ~isfile(reportPath)
        error('Report file not found: %s', reportPath);
    end
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    txt = fileread(reportPath);

    % Parse report tables.
    mrrTbl       = parseMetricTable(txt, '**MRR@k (unique laws)**');
    mrrTbl.idf_svd.label = " IDF--SVD";
    mrrTbl.kahm_query_mb_corpus.label = " KAHM";
    mrrTbl.mixedbread_true.label = " Mixedbread";
    hitTbl       = parseMetricTable(txt, '**Hit@k**');
    hitTbl.idf_svd.label = "IDF--SVD";
    hitTbl.kahm_query_mb_corpus.label = "KAHM";
    hitTbl.mixedbread_true.label = "Mixedbread";
    top1Tbl      = parseMetricTable(txt, '**Top-1 accuracy**');
    top1Tbl.idf_svd.label = "IDF--SVD";
    top1Tbl.kahm_query_mb_corpus.label = "KAHM";
    top1Tbl.mixedbread_true.label = "Mixedbread";
    majTbl       = parseMetricTable(txt, '**Majority-accuracy**');
    majTbl.idf_svd.label = "IDF--SVD";
    majTbl.kahm_query_mb_corpus.label = "KAHM";
    majTbl.mixedbread_true.label = "Mixedbread";
    consTbl      = parseMetricTable(txt, '**Mean consensus fraction**');
    consTbl.idf_svd.label = "IDF--SVD";
    consTbl.kahm_query_mb_corpus.label = "KAHM";
    consTbl.mixedbread_true.label = "Mixedbread";
    liftTbl      = parseMetricTable(txt, '**Mean lift (prior)**');
    liftTbl.idf_svd.label = "IDF--SVD";
    liftTbl.kahm_query_mb_corpus.label = "KAHM";
    liftTbl.mixedbread_true.label = "Mixedbread";
    runtimeTbl   = parseRuntimeTable(txt);

    % Consistent plotting style.
    set(groot, 'defaultAxesFontName', 'Helvetica');
    set(groot, 'defaultTextFontName', 'Helvetica');
    set(groot, 'defaultAxesFontSize', 18);
    set(groot, 'defaultTextInterpreter', 'tex');
    set(groot, 'defaultLegendInterpreter', 'tex');

    plotQualityFigure(mrrTbl, hitTbl, top1Tbl, outDir);
    plotConsensusFigure(majTbl, consTbl, liftTbl, outDir);
    plotTradeoffFigure(mrrTbl, runtimeTbl, outDir);
    writeLatexSnippet(outDir);
    fprintf('Generated MATLAB figure bundle in: %s\n', outDir);
end

function tbl = parseMetricTable(txt, label)
    [header, rows] = parseMarkdownTable(extractTableAfterLabel(txt, label));
    methods = header(2:end);
    tbl = struct();
    tbl.k = zeros(numel(rows), 1);
    for i = 1:numel(methods)
        fld = methodKey(methods{i});
        tbl.(fld).point = zeros(numel(rows), 1);
        tbl.(fld).lo = zeros(numel(rows), 1);
        tbl.(fld).hi = zeros(numel(rows), 1);
        tbl.(fld).label = methods{i};
    end
    for r = 1:numel(rows)
        row = rows{r};
        tbl.k(r) = str2double(strtrim(row{1}));
        for c = 2:numel(header)
            fld = methodKey(header{c});
            [pt, lo, hi] = parseCICell(row{c});
            tbl.(fld).point(r) = pt;
            tbl.(fld).lo(r) = lo;
            tbl.(fld).hi(r) = hi;
        end
    end
end



function tbl = parseRuntimeTable(txt)
    [~, rows] = parseMarkdownTable(extractTableAfterLabel(txt, '### Per-query online path comparison'));
    tbl = struct();
    for r = 1:numel(rows)
        row = rows{r};
        fld = methodKey(row{1});
        tbl.(fld).label = row{1};
        tbl.(fld).embed_ms = parseMsCell(row{3});
        tbl.(fld).search_ms = parseMsCell(row{4});
        tbl.(fld).total_ms = parseMsCell(row{5});
    end
end

function plotQualityFigure(mrrTbl, hitTbl, top1Tbl, outDir)
    fig = figure('Color', 'w', 'Position', [100 100 1200 360]);
    tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    ax1 = nexttile(tl, 1); hold(ax1, 'on');
    plotMetricWithCI(ax1, mrrTbl, 'MRR@k (unique laws)');
    xlabel(ax1, 'Retrieval cutoff k','FontSize',18); ylabel(ax1, 'MRR@k (unique laws)','FontSize',18);
    title(ax1, 'Law-level ranking quality','FontSize',18); grid(ax1, 'on'); box(ax1, 'on');

    ax2 = nexttile(tl, 2); hold(ax2, 'on');
    plotMetricWithCI(ax2, hitTbl, 'Hit@k');
    xlabel(ax2, 'Retrieval cutoff k','FontSize',18); ylabel(ax2, 'Hit@k','FontSize',18);
    title(ax2, 'Recall within top-k','FontSize',18); grid(ax2, 'on'); box(ax2, 'on');

    ax3 = nexttile(tl, 3); hold(ax3, 'on');
    plotMetricWithCI(ax3, top1Tbl, 'Top-1 accuracy');
    xlabel(ax3, 'Retrieval cutoff k','FontSize',18); ylabel(ax3, 'Top-1 accuracy','FontSize',18);
    title(ax3, 'Strict rank-1 accuracy','FontSize',18); grid(ax3, 'on'); box(ax3, 'on');

    lgd = legend(ax3, 'Location', 'southoutside', 'Orientation', 'horizontal','FontSize',18);
    lgd.Layout.Tile = 'south';

    exportFigure(fig, fullfile(outDir, 'quality_vs_k'));
    close(fig);
end

function plotConsensusFigure(majTbl, consTbl, liftTbl, outDir)
    fig = figure('Color', 'w', 'Position', [100 100 1200 360]);
    tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    ax1 = nexttile(tl, 1); hold(ax1, 'on');
    plotMetricWithCI(ax1, majTbl, 'Majority-accuracy');
    xlabel(ax1, 'Retrieval cutoff k','FontSize',18); ylabel(ax1, 'Majority-accuracy','FontSize',18);
    title(ax1, 'Vote-based correctness','FontSize',18); grid(ax1, 'on'); box(ax1, 'on');

    ax2 = nexttile(tl, 2); hold(ax2, 'on');
    plotMetricWithCI(ax2, consTbl, 'Mean consensus fraction');
    xlabel(ax2, 'Retrieval cutoff k','FontSize',18); ylabel(ax2, 'Mean consensus fraction','FontSize',18);
    title(ax2, 'Neighborhood purity','FontSize',18); grid(ax2, 'on'); box(ax2, 'on');

    ax3 = nexttile(tl, 3); hold(ax3, 'on');
    plotMetricWithCI(ax3, liftTbl, 'Mean lift (prior)');
    xlabel(ax3, 'Retrieval cutoff k','FontSize',18); ylabel(ax3, 'Mean lift (prior)','FontSize',18);
    title(ax3, 'Enrichment over corpus prior','FontSize',18); grid(ax3, 'on'); box(ax3, 'on');

    lgd = legend(ax3, 'Location', 'southoutside', 'Orientation', 'horizontal','FontSize',18);
    lgd.Layout.Tile = 'south';

    exportFigure(fig, fullfile(outDir, 'consensus_vs_k'));
    close(fig);
end

function plotTradeoffFigure(mrrTbl, runtimeTbl, outDir)
    fig = figure('Color', 'w', 'Position', [100 100 1000 800]);
    ax = axes(fig); hold(ax, 'on');

    methods = {'idf_svd','kahm_query_mb_corpus','mixedbread_true'};
    labels = {mrrTbl.idf_svd.label, mrrTbl.kahm_query_mb_corpus.label, mrrTbl.mixedbread_true.label};
    colors = methodColors();
    markers = methodMarkers();

    idx20 = find(mrrTbl.k == 20, 1, 'first');
    if isempty(idx20)
        error('Could not find k=20 in MRR table.');
    end

    xs = [runtimeTbl.idf_svd.total_ms, runtimeTbl.kahm_query_mb_corpus.total_ms, runtimeTbl.mixedbread_true.total_ms];
    ys = [mrrTbl.idf_svd.point(idx20), mrrTbl.kahm_query_mb_corpus.point(idx20), mrrTbl.mixedbread_true.point(idx20)];
    ylo = [mrrTbl.idf_svd.lo(idx20), mrrTbl.kahm_query_mb_corpus.lo(idx20), mrrTbl.mixedbread_true.lo(idx20)];
    yhi = [mrrTbl.idf_svd.hi(idx20), mrrTbl.kahm_query_mb_corpus.hi(idx20), mrrTbl.mixedbread_true.hi(idx20)];

    for i = 1:numel(methods)
        m = methods{i};
        errorbar(ax, xs(i), ys(i), ys(i)-ylo(i), yhi(i)-ys(i), ...
            'LineStyle', 'none', 'Color', colors.(m), 'LineWidth', 1.2, 'CapSize', 8);
        scatter(ax, xs(i), ys(i), 80, 'Marker', markers.(m), ...
            'MarkerFaceColor', colors.(m), 'MarkerEdgeColor', colors.(m), 'DisplayName', labels{i});
        text(ax, xs(i)*1.03, ys(i), labels{i}, 'Color', colors.(m), 'FontSize', 18);
    end

    set(ax, 'XScale', 'log');
    xlabel(ax, 'Online per-query time (ms, log scale)','FontSize',18);
    ylabel(ax, 'MRR@20 (unique laws)','FontSize',18);
    title(ax, 'Compute-quality trade-off','FontSize',18);
    grid(ax, 'on'); box(ax, 'on');

    exportFigure(fig, fullfile(outDir, 'compute_quality_tradeoff'));
    close(fig);
end



function plotMetricWithCI(ax, tbl, ~)
    methods = {'idf_svd','kahm_query_mb_corpus','mixedbread_true'};
    colors = methodColors();
    markers = methodMarkers();
    ks = tbl.k(:)';
    for i = 1:numel(methods)
        m = methods{i};
        pt = tbl.(m).point(:)';
        lo = tbl.(m).lo(:)';
        hi = tbl.(m).hi(:)';
        c = colors.(m);
        patch(ax, [ks fliplr(ks)], [lo fliplr(hi)], c, ...
            'FaceAlpha', 0.14, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        plot(ax, ks, pt, 'Color', c, 'LineWidth', 1.8, 'Marker', markers.(m), ...
            'MarkerSize', 6, 'DisplayName', tbl.(m).label);
    end
    xlim(ax, [min(ks)-0.5, max(ks)+0.5]);
end



function exportFigure(fig, basePath)
    drawnow;
    epsPath = [basePath '.eps'];
    if exist('exportgraphics', 'file') == 2
        exportgraphics(fig, epsPath, 'ContentType', 'vector');
    else
        print(fig, epsPath, '-depsc', '-vector');
    end
end

function writeLatexSnippet(outDir)
    p = fullfile(outDir, 'figures_matlab_latex.tex');
    fid = fopen(p, 'w');
    if fid < 0
        error('Could not open LaTeX snippet for writing: %s', p);
    end
    cleaner = onCleanup(@() fclose(fid)); 

    fprintf(fid, '%% ---------------------------------------------------------------\n');
    fprintf(fid, '%% Suggested MATLAB-generated figure insertions for Section 4.5\n');
    fprintf(fid, '%% Requires: \\usepackage{graphicx}\n');
    fprintf(fid, '%% ---------------------------------------------------------------\n\n');

    fprintf(fid, '\\begin{figure*}[t]\n');
    fprintf(fid, '    \\centering\n');
    fprintf(fid, '    \\includegraphics[width=0.92\\textwidth]{figures/quality_vs_k.pdf}\n');
    fprintf(fid, '    \\caption{Main retrieval-quality metrics across cutoffs. The proposed KAHM (query$\\rightarrow$MB corpus) encoder consistently dominates the lexical IDF--SVD baseline and remains above the direct transformer-query baseline on the principal rank-sensitive measures. Shaded regions indicate paired-bootstrap 95\\%% confidence intervals.}\n');
    fprintf(fid, '    \\label{fig:quality-vs-k}\n');
    fprintf(fid, '\\end{figure*}\n\n');

    fprintf(fid, '\\begin{figure*}[t]\n');
    fprintf(fid, '    \\centering\n');
    fprintf(fid, '    \\includegraphics[width=0.92\\textwidth]{figures/consensus_vs_k.pdf}\n');
    fprintf(fid, '    \\caption{Consensus- and routing-sensitive metrics across cutoffs. The KAHM-based encoder yields consistently purer retrieval neighborhoods around the gold law, as reflected in majority-accuracy, mean consensus fraction, and mean lift. Shaded regions indicate paired-bootstrap 95\\%% confidence intervals.}\n');
    fprintf(fid, '    \\label{fig:consensus-vs-k}\n');
    fprintf(fid, '\\end{figure*}\n\n');

    fprintf(fid, '\\begin{figure*}[t]\n');
    fprintf(fid, '    \\centering\n');
    fprintf(fid, '    \\includegraphics[width=0.72\\textwidth]{figures/compute_quality_tradeoff.pdf}\n');
    fprintf(fid, '    \\caption{Compute-quality trade-off based on online per-query time and MRR@20. The proposed KAHM-based encoder occupies the middle ground between the very fast lexical baseline and the expensive online transformer baseline, while attaining the strongest retrieval quality among the three methods in this run. Error bars show paired-bootstrap 95\\%% confidence intervals for MRR@20.}\n');
    fprintf(fid, '    \\label{fig:compute-quality-tradeoff}\n');
    fprintf(fid, '\\end{figure*}\n\n');

end

function [header, rows] = parseMarkdownTable(tableLines)
    n = numel(tableLines);
    if n < 3
        error('Markdown table appears incomplete.');
    end
    header = splitMarkdownRow(tableLines{1});
    rows = cell(n-2, 1);
    rr = 0;
    for i = 3:n
        row = splitMarkdownRow(tableLines{i});
        if numel(row) == numel(header)
            rr = rr + 1;
            rows{rr} = row;
        end
    end
    rows = rows(1:rr);
end

function parts = splitMarkdownRow(line)
    line = strtrim(line);
    if startsWith(line, '|')
        line = extractAfter(line, 1);
    end
    if endsWith(line, '|')
        line = extractBefore(line, strlength(line));
    end
    cells = regexp(char(line), '\s*\|\s*', 'split');
    parts = cellfun(@strtrim, cells, 'UniformOutput', false);
end

function lines = extractTableAfterLabel(txt, label)
    pos = strfind(txt, label);
    if isempty(pos)
        error('Label not found: %s', label);
    end
    sub = txt(pos(1) + strlength(label):end);
    rawLines = regexp(sub, '\r?\n', 'split');
    lines = {};
    started = false;
    for i = 1:numel(rawLines)
        ln = strtrim(rawLines{i});
        if startsWith(ln, '|')
            lines{end+1,1} = ln; %#ok<AGROW>
            started = true;
        elseif started
            break;
        end
    end
    if isempty(lines)
        error('No markdown table found after label: %s', label);
    end
end

function [pt, lo, hi] = parseCICell(cellStr)
    expr = '([+-]?\d+(?:\.\d+)?)\s*\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]';
    tok = regexp(cellStr, expr, 'tokens', 'once');
    if isempty(tok)
        error('Could not parse CI cell: %s', cellStr);
    end
    pt = str2double(tok{1});
    lo = str2double(tok{2});
    hi = str2double(tok{3});
end

function ms = parseMsCell(cellStr)
    tok = regexp(cellStr, '([+-]?\d+(?:\.\d+)?)\s*ms', 'tokens', 'once');
    if isempty(tok)
        error('Could not parse milliseconds from cell: %s', cellStr);
    end
    ms = str2double(tok{1});
end

function key = methodKey(label)
    s = lower(strtrim(label));
    s = strrep(s, '–', '-');
    s = strrep(s, '—', '-');
    s = regexprep(s, '[^a-z0-9]+', '_');
    s = regexprep(s, '^_+|_+$', '');
    if strcmp(s, 'mixedbread_true_reference')
        s = 'mixedbread_true';
    end
    if strcmp(s, 'kahm_query_mb_corpus_gradient_free_query_adapter_idf_svd_features_mapped_into_the_transformer_embedding_space_frozen_transformer_corpus_embeddings')
        s = 'kahm_query_mb_corpus';
    end
    key = s;
end

function colors = methodColors()
    colors.idf_svd = [0.1216 0.4667 0.7059];
    colors.kahm_query_mb_corpus = [0.1725 0.6275 0.1725];
    colors.mixedbread_true = [1.0000 0.4980 0.0549];
end

function markers = methodMarkers()
    markers.idf_svd = 'o';
    markers.kahm_query_mb_corpus = 's';
    markers.mixedbread_true = '^';
end
