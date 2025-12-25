function [predicted_labels, decision_values] = applyRLDAClassifier(features, classifier)
% applyRLDAClassifier - 应用RLDA分类器到特征矩阵
%
% 输入:
%   features   - 特征矩阵 (特征数 x 样本数)
%   classifier - 包含w、b字段的分类器结构体
%
% 输出:
%   predicted_labels - 预测的类别标签 (1或2)
%   decision_values  - 判别值 (未经过阈值处理的原始分数)

    % 确保输入格式正确
    if isstruct(features) && isfield(features, 'x')
        % 如果输入是结构体，提取特征矩阵
        features = features.x;
    end
    
    % 应用线性分类器
    % w的维度应该是 [特征数 x 1]
    % 特征矩阵应该是 [特征数 x 样本数]
    decision_values = classifier.w' * features + classifier.b;
    
    % 转换为类别标签 (大于0为类别2，否则为类别1)
    predicted_labels = double(decision_values > 0) + 1;
    
    % 转置结果使其成为行向量 (如果需要)
    decision_values = decision_values(:)';
    predicted_labels = predicted_labels(:)';
end